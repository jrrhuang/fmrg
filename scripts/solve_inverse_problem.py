#!/usr/bin/env python3
"""
FMRG inverse-problem solver on FLUX FlowMap.

Supports FMRG-J and FMRG-E variants of the proposed method via --grad_mode,
plus FlowDPS and FlowChef baselines via --method. Processes one image at a
time or a directory of images in batches.

Example:
    python scripts/solve_inverse_problem.py \\
        --task_config configs/inverse_problems/sr_config.yaml \\
        --image_path /path/to/images/ \\
        --method fmrg --grad_mode jac --normalize_grad \\
        --num_steps 30 --step_size 5.0 --num_optim_iters 1 \\
        --sample_mode flow_map1 --loss_mode pixel \\
        --prompt "a photo of a dog" --seed 0
"""

import argparse
import glob
import json
import os
import signal
import shutil
import sys
import sys, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from pathlib import Path

import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import yaml
import numpy as np


# Global interrupt state for signal handling
class _InterruptState:
    interrupted = False

_interrupt_state = _InterruptState()


def _signal_handler(signum, frame):
    """Handle SIGUSR1 (SLURM timeout warning) and SIGTERM (preemption) gracefully."""
    _interrupt_state.interrupted = True
    sig_name = signal.Signals(signum).name
    print(f"\n[SIGNAL] Received {sig_name} - will finish current batch then exit for requeue.")

# =============================================================================
# CONFIGURABLE PATHS - Modify these for your setup
# =============================================================================

# Directory containing this script
SCRIPT_DIR = Path(__file__).parent.absolute()

# Default output directory for results
DEFAULT_SAVE_DIR = "./results"

# Default paths to FlowMap LoRA weights by resolution
_REPO = SCRIPT_DIR.parent
LORA_PATHS = {
    256: os.environ.get("FMRG_LORA_PATH_256", os.environ.get("FMRG_LORA_PATH", str(_REPO / "checkpoints/flux-flowmap-lora"))),
    512: os.environ.get("FMRG_LORA_PATH_512", str(_REPO / "checkpoints/flux-flowmap-lora-512")),
}
SUPPORTED_RESOLUTIONS = list(LORA_PATHS.keys())

# Default FLUX model ID (can override via --model_id)
DEFAULT_MODEL_ID = "black-forest-labs/FLUX.1-dev"

# =============================================================================
# Imports (no need to modify below this line)
# =============================================================================

from functions.degradation import get_degradation
from munch import munchify
from fluxfm_sampler_ip import FluxFlowMapSampler, FluxFlowDPS, FluxFlowChef


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_metrics(recon_path: str, label_path: str, device: str = 'cuda') -> dict:
    """
    Compute PSNR, SSIM, and LPIPS between reconstruction and ground truth.
    Matches FlowDPS eval.py computation exactly.
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr
    from pytorch_msssim import ssim
    import lpips

    # Load images
    recon_img = Image.open(recon_path).convert('RGB')
    label_img = Image.open(label_path).convert('RGB')

    # Resize label to match recon if needed (for skip_label_save case)
    if label_img.size != recon_img.size:
        label_img = label_img.resize(recon_img.size, Image.BICUBIC)

    transform_tensor = transforms.ToTensor()
    recon_np = np.array(transform_tensor(recon_img)) * 255
    label_np = np.array(transform_tensor(label_img)) * 255
    psnr_val = psnr(label_np, recon_np, data_range=255)

    recon_tensor = transform_tensor(recon_img).unsqueeze(0) * 255
    label_tensor = transform_tensor(label_img).unsqueeze(0) * 255
    ssim_val = ssim(label_tensor, recon_tensor).item()

    lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    lpips_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    recon_lpips = lpips_transform(recon_img).to(device)
    label_lpips = lpips_transform(label_img).to(device)
    lpips_val = lpips_fn(label_lpips, recon_lpips).item()

    return {
        'psnr': float(psnr_val),
        'ssim': float(ssim_val),
        'lpips': float(lpips_val)
    }


def main():
    parser = argparse.ArgumentParser(description="FLUX FlowMap inverse problem solver (FMRG, FlowDPS, FlowChef)")

    # Required
    parser.add_argument('--task_config', type=str, required=True, help='Task config YAML')
    parser.add_argument('--image_path', type=str, required=True, help='Input image path or directory')
    parser.add_argument('--split_idx', type=int, default=0, help='Split index for parallel processing')
    parser.add_argument('--num_splits', type=int, default=1, help='Number of splits (1=no splitting)')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for processing multiple images (default: 2 for 256 resolution on L40S)')

    # Sampling parameters
    parser.add_argument('--num_steps', type=int, default=28, help='Number of sampling steps')
    parser.add_argument('--guidance_scale', type=float, default=3.5,
                        help='FLUX embedded CFG scale (distinct from FMRG guidance strength step_size)')
    parser.add_argument('--step_size', type=float, default=7.0,
                        help='Guidance strength lambda')
    parser.add_argument('--num_optim_iters', type=int, default=10,
                        help='Inner gradient steps n_opt per interval')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--early_stop', type=int, default=0,
                        help='Early stopping (0=off, 1=on). When on, applies guidance for 2/3 of steps then completes via one uncontrolled flow-map step to t=0 (2/3-then-jump-to-0).')

    parser.add_argument('--method', type=str, default='fmrg', choices=['fmrg', 'flowdps', 'flowchef'],
                        help='Sampling method: fmrg (Flow Map Reward Guidance, ours), flowdps (FlowDPS baseline), flowchef (FlowChef baseline)')

    parser.add_argument('--grad_mode', type=str, default='jac', choices=['jac', 'euc'],
                        help='FMRG variant: jac (FMRG-J) or euc (FMRG-E)')
    parser.add_argument('--normalize_grad', action='store_true',
                        help='Velocity-norm gradient rescaling')
    parser.add_argument('--sample_mode', type=str, default='flow_map1', choices=['flow_map1', 'flow_map2', 'flow_matching'],
                        help='How the flow map is used to advance the trajectory: '
                             'flow_map2 (canonical 2-NFE/step), '
                             'flow_map1 (1-NFE/step linear-interp shortcut), '
                             'flow_matching (1-NFE Euler step; FlowDPS/FlowChef baselines)')
    parser.add_argument('--loss_mode', type=str, default='latent', choices=['latent', 'pixel'],
                        help='Reward-loss computation space')
    parser.add_argument('--loss_func', type=str, default='norm', choices=['norm', 'sum'],
                        help='Reduction over the residual ‖A·D(z) − y‖')
    parser.add_argument('--enable_callback', action='store_true',
                        help='Enable callback for progress tracking (slower due to VAE decode each step)')

    # Model paths
    parser.add_argument('--prompt', type=str, default='',
                        help='Text prompt (used for all images unless --prompts_file is set)')
    parser.add_argument('--prompts_file', type=str, default=None,
                        help='Path to prompts.json mapping image filenames to per-image prompts (for class-conditional)')
    parser.add_argument('--lora_path', type=str, default=None,
                        help='Path to FlowMap LoRA (auto-selected based on resolution if not specified)')
    parser.add_argument('--model_id', type=str, default=DEFAULT_MODEL_ID,
                        help='FLUX model ID')

    # Output
    parser.add_argument('--save_dir', type=str, default=DEFAULT_SAVE_DIR, help='Output directory')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    parser.add_argument('--resolution', type=int, default=256,
                        help=f'Image resolution (only {SUPPORTED_RESOLUTIONS} supported)')
    parser.add_argument('--compute_metrics', action='store_true', help='Compute PSNR, SSIM, LPIPS metrics')
    parser.add_argument('--skip_input_save', action='store_true', help='Skip saving input/measurement images')
    parser.add_argument('--skip_label_save', action='store_true', help='Skip saving label/ground truth images (metrics computed from original path)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume mode: skip images with existing recon, preserve progress dirs')

    args = parser.parse_args()

    # Register signal handlers for graceful interruption (SLURM preemption/timeout)
    signal.signal(signal.SIGUSR1, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    # Validate resolution
    if args.resolution not in SUPPORTED_RESOLUTIONS:
        parser.error(f"Resolution must be one of {SUPPORTED_RESOLUTIONS}, got {args.resolution}")

    # Auto-select LoRA path based on resolution if not specified
    if args.lora_path is None:
        args.lora_path = LORA_PATHS[args.resolution]
        print(f"Auto-selected LoRA for resolution {args.resolution}: {args.lora_path}")

    # Set random seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # For multi-GPU

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load task config
    task_config = load_yaml(args.task_config)
    measure_config = task_config['measurement']
    operator_name = measure_config['operator']['name']

    # Get deg_scale from config (default 4 for SR, 1 for others)
    deg_scale = measure_config['operator'].get('deg_scale', 4)

    # Map task names to FlowDPS degradation names and internal task names
    # FlowDPS uses: sr_avgpool, sr_bicubic, deblur_gauss, deblur_motion
    task_name_map = {
        'super_resolution': ('sr_avgpool', 'sr'),
        'sr_avgpool': ('sr_avgpool', 'sr'),
        'sr_bicubic': ('sr_bicubic', 'sr'),
        'gaussian_blur': ('deblur_gauss', 'deblur'),
        'deblur_gauss': ('deblur_gauss', 'deblur'),
        'motion_blur': ('deblur_motion', 'deblur'),
        'deblur_motion': ('deblur_motion', 'deblur'),
        'inpainting': ('box_inpainting', 'inpaint'),
        'box_inpainting': ('box_inpainting', 'inpaint'),
    }

    if operator_name not in task_name_map:
        raise ValueError(f"Unknown operator: {operator_name}. Supported: {list(task_name_map.keys())}")

    degradation_name, task_name = task_name_map[operator_name]
    print(f"Task: {task_name} (degradation: {degradation_name})")

    deg_config = munchify({
        'channels': 3,
        'image_size': args.resolution,
        'deg_scale': deg_scale,
    })
    operator = get_degradation(degradation_name, deg_config, device)

    noise_sigma = measure_config['noise'].get('sigma', 0.03)

    # Get list of image paths
    image_path = Path(args.image_path)
    if image_path.is_dir():
        # Directory: get all images
        image_paths = sorted(
            glob.glob(str(image_path / "*.png")) +
            glob.glob(str(image_path / "*.jpg")) +
            glob.glob(str(image_path / "*.jpeg"))
        )
        if not image_paths:
            raise ValueError(f"No images found in {image_path}")
        print(f"Found {len(image_paths)} images in {image_path}")

        # Apply splitting if requested
        if args.num_splits > 1:
            total = len(image_paths)
            split_size = (total + args.num_splits - 1) // args.num_splits
            start_idx = args.split_idx * split_size
            end_idx = min(start_idx + split_size, total)
            image_paths = image_paths[start_idx:end_idx]
            print(f"Split {args.split_idx + 1}/{args.num_splits}: images {start_idx}-{end_idx-1} ({len(image_paths)} images)")
    else:
        # Single image
        image_paths = [str(image_path)]

    # Resume mode: skip already-completed images
    if args.resume:
        recon_dir = Path(args.save_dir) / measure_config['operator']['name'] / 'recon'
        original_count = len(image_paths)
        image_paths = [
            p for p in image_paths
            if not (recon_dir / (Path(p).stem + '.png')).exists()
        ]
        skipped = original_count - len(image_paths)
        if skipped > 0:
            print(f"Resume mode: skipping {skipped}/{original_count} already-completed images")
        if len(image_paths) == 0:
            print("All images already completed. Nothing to do.")
            sys.exit(0)

    # Load and preprocess image transform
    transform = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    # Load per-image prompts if provided
    per_image_prompts = None
    if args.prompts_file:
        with open(args.prompts_file) as f:
            per_image_prompts = json.load(f)
        all_prompts = list(set(per_image_prompts.values()))
        print(f"Loaded {len(per_image_prompts)} per-image prompts ({len(all_prompts)} unique)")
        init_prompt = all_prompts
    else:
        init_prompt = args.prompt

    sampler_cls = {
        'fmrg': FluxFlowMapSampler,
        'flowdps': FluxFlowDPS,
        'flowchef': FluxFlowChef,
    }[args.method]
    print(f"Loading {sampler_cls.__name__} (method={args.method})...")
    sampler = sampler_cls(
        model_id=args.model_id,
        lora_path=args.lora_path,
        prompt=init_prompt,
    )

    # Setup output directories
    out_path = Path(args.save_dir) / measure_config['operator']['name']
    dirs_to_create = ['recon', 'progress']
    if not args.skip_input_save:
        dirs_to_create.append('input')
    if not args.skip_label_save:
        dirs_to_create.append('label')
    for d in dirs_to_create:
        (out_path / d).mkdir(parents=True, exist_ok=True)

    print(f"Running FMRG-{'J' if args.grad_mode == 'jac' else 'E'} "
          f"({args.sample_mode}, loss={args.loss_mode}, normalize_grad={args.normalize_grad})...")
    print(f"Batch size: {args.batch_size}")

    num_batches = (len(image_paths) + args.batch_size - 1) // args.batch_size
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * args.batch_size
        batch_end = min(batch_start + args.batch_size, len(image_paths))
        batch_paths = [Path(p) for p in image_paths[batch_start:batch_end]]
        actual_batch_size = len(batch_paths)

        # Load batch of images
        batch_imgs = []
        batch_fnames = []
        for img_path in batch_paths:
            img = Image.open(img_path).convert('RGB')
            ref_img = transform(img).to(device)
            # Convert to [-1, 1] range (FLUX VAE convention)
            ref_img = ref_img * 2 - 1
            batch_imgs.append(ref_img)
            batch_fnames.append(img_path.stem + '.png')

        # Stack into batch tensor [N, 3, H, W]
        ref_imgs = torch.stack(batch_imgs, dim=0)

        # Generate batched measurement using operator.A()
        y = operator.A(ref_imgs)
        y_n = y + noise_sigma * torch.randn_like(y)

        # Setup progress tracking for batch
        # For batched callback, we save all images in the batch
        loss_histories = [[] for _ in range(actual_batch_size)]
        progress_dirs = []
        for img_path in batch_paths:
            progress_dir = out_path / 'progress' / img_path.stem
            if not args.resume and progress_dir.exists():
                shutil.rmtree(progress_dir)
            progress_dir.mkdir(parents=True, exist_ok=True)
            progress_dirs.append(progress_dir)

        def save_progress_callback_batch(step, imgs, t, loss):
            # imgs is [N, 3, H, W], save each image separately
            for i in range(imgs.shape[0]):
                loss_histories[i].append({'step': step, 't': t, 'loss': loss})
                save_image(imgs[i:i+1], progress_dirs[i] / f'step_{step:03d}_t_{t:.3f}_loss_{loss:.1f}.png', normalize=True)
        callback = save_progress_callback_batch

        # Set per-image prompt embeddings for this batch
        if per_image_prompts is not None:
            batch_pe = []
            batch_ppe = []
            for fname in batch_fnames:
                p = per_image_prompts.get(fname, args.prompt)
                pe, ppe = sampler.prompt_embeddings[p]
                batch_pe.append(pe)
                batch_ppe.append(ppe)
            sampler.cached_prompt_embeds = torch.cat(batch_pe, dim=0)
            sampler.cached_pooled_prompt_embeds = torch.cat(batch_ppe, dim=0)

        if args.method == 'fmrg':
            results = sampler.sample_inverse_problem(
                measurement=y_n,
                operator=operator,
                task=task_name,
                num_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                step_size=args.step_size,
                img_shape=(args.resolution, args.resolution),
                grad_mode=args.grad_mode,
                sample_mode=args.sample_mode,
                loss_mode=args.loss_mode,
                normalize_grad=args.normalize_grad,
                loss_func=args.loss_func,
                num_optim_iters=args.num_optim_iters,
                seed=args.seed,
                callback=callback,
                early_stop=args.early_stop,
                enable_callback=args.enable_callback,
            )
        elif args.method == 'flowdps':
            results = sampler.sample_flowdps(
                measurement=y_n,
                operator=operator,
                task=task_name,
                NFE=args.num_steps,
                guidance_scale=args.guidance_scale,
                step_size=args.step_size,
                num_optim_iters=args.num_optim_iters,
                img_shape=(args.resolution, args.resolution),
                seed=args.seed,
                callback=callback,
            )
        elif args.method == 'flowchef':
            results = sampler.sample_flowchef(
                measurement=y_n,
                operator=operator,
                task=task_name,
                NFE=args.num_steps,
                guidance_scale=args.guidance_scale,
                step_size=args.step_size,
                num_optim_iters=args.num_optim_iters,
                img_shape=(args.resolution, args.resolution),
                seed=args.seed,
                callback=callback,
            )

        # Save outputs for each image in batch
        for i, (img_path, fname) in enumerate(zip(batch_paths, batch_fnames)):
            result = results[i:i+1]  # Keep batch dim for save_image
            ref_img = ref_imgs[i:i+1]
            y_n_i = y_n[i:i+1]

            if not args.skip_input_save:
                at_y = operator.At(y_n_i)
                # Motion-blur At() returns 4D with possibly off-by-one spatial size; SR's
                # SVD pinv returns a flat tensor. Match shape to ref_img either way.
                if at_y.dim() == 4 and at_y.shape[-2:] != ref_img.shape[-2:]:
                    at_y = torch.nn.functional.interpolate(
                        at_y, size=ref_img.shape[-2:], mode='bilinear', align_corners=False
                    )
                if at_y.numel() == ref_img.numel():
                    at_y = at_y.reshape(ref_img.shape)
                save_image(at_y, out_path / 'input' / fname, normalize=True)
            if not args.skip_label_save:
                save_image(ref_img, out_path / 'label' / fname, normalize=True)
            save_image(result, out_path / 'recon' / fname, normalize=True)

            # Compute metrics if requested
            metrics = None
            if args.compute_metrics:
                recon_path = str(out_path / 'recon' / fname)
                # Use original image path if label not saved, otherwise use saved label
                if args.skip_label_save:
                    label_path = str(img_path)
                else:
                    label_path = str(out_path / 'label' / fname)
                metrics = compute_metrics(recon_path, label_path, device=str(device))
                print(f"  {fname}: PSNR: {metrics['psnr']:.2f}, SSIM: {metrics['ssim']:.4f}, LPIPS: {metrics['lpips']:.4f}")

            # Save metadata (always save to recon directory)
            metadata = {
                'parameters': {
                    'task': task_name,
                    'num_steps': args.num_steps,
                    'guidance_scale': args.guidance_scale,
                    'step_size': args.step_size,
                    'seed': args.seed,
                    'prompt': args.prompt,
                    'resolution': args.resolution,
                    'model_id': args.model_id,
                    'lora_path': args.lora_path,
                    'batch_size': args.batch_size,
                    'grad_mode': args.grad_mode,
                    'normalize_grad': args.normalize_grad,
                    'sample_mode': args.sample_mode,
                    'loss_mode': args.loss_mode,
                    'num_optim_iters': args.num_optim_iters,
                    'early_stop': args.early_stop,
                },
                'final_loss': loss_histories[i][-1]['loss'] if loss_histories[i] else None,
                'loss_history': loss_histories[i],
            }
            if metrics:
                metadata['metrics'] = metrics

            with open(progress_dirs[i] / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        # Check for interrupt signal after completing batch
        if _interrupt_state.interrupted:
            print(f"\n[INTERRUPT] Signal received. Finished batch {batch_idx+1}/{num_batches}.")
            print(f"[INTERRUPT] Saved {batch_end}/{len(image_paths)} images. Exiting for requeue.")
            sys.exit(0)

    print(f"Done! Results saved to {out_path}")


if __name__ == '__main__':
    main()
