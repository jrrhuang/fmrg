"""
Generate GenEval images using FMRG reward guidance (FMRG-J / FMRG-E).

Produces output in GenEval format:
  {outdir}/{index:05d}/metadata.jsonl
  {outdir}/{index:05d}/samples/{seed:05d}.png

Supports sharding via --start_idx / --end_idx for parallel jobs.
Resumes automatically (skips existing images).

Usage:
    python generate_geneval.py --grad_mode jac --normalize_grad --nfe 8 --step_size 0.3 --num_optim_iters 1
    python generate_geneval.py --grad_mode euc --nfe 8 --step_size 1.0 --num_optim_iters 1 --early_stop 5
"""

import argparse
import json
import os
import sys
import time

import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

sys.path.insert(0, REPO_ROOT)

# Must be imported before fluxfm_sampler_reward (which lazy-imports ImageReward)
# to monkey-patch transformers 5.x APIs that ImageReward expects from 4.x.
import imagereward_compat  # noqa: F401

from fluxfm_sampler_reward import FluxFlowMapSampler, RewardEnsemble


def _nfe_to_steps(nfe: int, sample_mode: str) -> int:
    """flow_map1 uses 1 NFE per step; flow_map2 uses 2."""
    return nfe if sample_mode == 'flow_map1' else nfe // 2

LORA_PATH = os.environ.get("FMRG_LORA_PATH", os.path.join(REPO_ROOT, "checkpoints", "flux-flowmap-lora-512"))
GENEVAL_PROMPTS = os.path.join(REPO_ROOT, "data", "geneval_prompts", "evaluation_metadata.jsonl")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate GenEval images with FMRG reward guidance")

    # FMRG variant
    parser.add_argument("--grad_mode", type=str, required=True, choices=["jac", "euc"],
                        help="FMRG variant: jac (Jacobian) or euc (Euclidean)")
    parser.add_argument("--normalize_grad", action="store_true",
                        help="Rescale gradient to velocity norm")
    parser.add_argument("--sample_mode", type=str, default="flow_map2",
                        choices=["flow_map1", "flow_map2"],
                        help="flow_map2 (2 NFE/step) or flow_map1 (1 NFE/step)")
    parser.add_argument("--nfe", type=int, default=8)
    parser.add_argument("--step_size", type=float, default=1.0)
    parser.add_argument("--num_optim_iters", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--grad_checkpointing", action="store_true")
    parser.add_argument("--warmup_steps", type=int, default=0,
                        help="Noise optimization steps before guided sampling (0=disabled)")
    parser.add_argument("--warmup_particles", type=int, default=1,
                        help="Number of particles to sample during warm-up")
    parser.add_argument("--warmup_lr", type=float, default=None,
                        help="Learning rate for warm-up (default: step_size)")

    # Reward ensemble weights
    parser.add_argument("--hps_weight", type=float, default=5.0)
    parser.add_argument("--imagereward_weight", type=float, default=1.0)
    parser.add_argument("--pickscore_weight", type=float, default=0.05)
    parser.add_argument("--clip_weight", type=float, default=0.01)

    # Paths
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (auto-generated if None)")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Path to evaluation_metadata.jsonl (default: GenEval standard)")
    parser.add_argument("--cache_dir", type=str,
                        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))

    # Sharding
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)

    # Early stop
    parser.add_argument("--early_stop", type=int, default=0,
                        help="Early stop step (0=disabled)")
    parser.add_argument("--unguided_steps", type=int, default=0,
                        help="Number of unguided steps after early stop before final decode (0=default single jump)")

    # Samples per prompt (GenEval uses 4 seeds: 0-3)
    parser.add_argument("--num_samples", type=int, default=1,
                        help="Number of samples per prompt (default: 1, use seed for different samples)")

    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    sample_mode = args.sample_mode
    grad_norm_mode_internal = 'normalize' if args.normalize_grad else 'none'
    steps = _nfe_to_steps(args.nfe, sample_mode)
    early_stop = args.early_stop

    # Auto-generate output dir
    if args.output_dir is None:
        settings = f"fmrg_{args.grad_mode}_nfe{args.nfe}_ss{args.step_size}_oi{args.num_optim_iters}"
        args.output_dir = os.path.join(SCRIPT_DIR, "outputs", "geneval", settings)

    print(f"FMRG-{'J' if args.grad_mode == 'jac' else 'E'} GenEval generation")
    print(f"  sample_mode={sample_mode}, normalize_grad={args.normalize_grad}")
    print(f"  nfe={args.nfe}, steps={steps}, early_stop={early_stop}")
    print(f"  step_size={args.step_size}, num_optim_iters={args.num_optim_iters}")
    print(f"  seed={args.seed}, resolution={args.resolution}")
    print(f"  grad_checkpointing={args.grad_checkpointing}")
    print(f"Output: {args.output_dir}")

    # Load GenEval prompts
    prompts_file = args.prompts_file or GENEVAL_PROMPTS
    with open(prompts_file) as f:
        metadatas = [json.loads(line) for line in f]

    start_idx = args.start_idx
    end_idx = args.end_idx if args.end_idx > 0 else len(metadatas)
    end_idx = min(end_idx, len(metadatas))
    print(f"Prompts: {start_idx} to {end_idx} (of {len(metadatas)})")

    # Collect unique prompts for this shard
    shard_prompts = []
    for idx in range(start_idx, end_idx):
        prompt = metadatas[idx]["prompt"]
        if prompt not in shard_prompts:
            shard_prompts.append(prompt)

    print(f"\nInitializing FluxFlowMapSampler with {len(shard_prompts)} prompts...")
    sampler = FluxFlowMapSampler(
        prompt=shard_prompts[0],
        additional_prompts=shard_prompts[1:] if len(shard_prompts) > 1 else None,
        lora_path=LORA_PATH,
        device=device,
    )

    resolution = args.resolution
    guidance_scale = 3.5

    # Load reward ensemble
    print("Loading reward ensemble...")
    reward_ensemble = RewardEnsemble(
        device=device, dtype=torch.float32, cache_dir=args.cache_dir,
        hps_weight=args.hps_weight, imagereward_weight=args.imagereward_weight,
        pickscore_weight=args.pickscore_weight, clip_weight=args.clip_weight,
    )

    # Generate images
    total = end_idx - start_idx
    completed = 0
    skipped = 0
    t_start = time.time()

    for idx in range(start_idx, end_idx):
        metadata = metadatas[idx]
        prompt = metadata["prompt"]
        outpath = os.path.join(args.output_dir, f"{idx:05d}")
        img_path = os.path.join(outpath, "samples", f"{args.seed:05d}.png")

        # Resume: skip existing (check size > 0 to catch corrupt files from killed jobs)
        if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
            skipped += 1
            continue

        # Switch prompt embeddings (also updates prompt_text for reward model)
        sampler.set_prompt_embeddings(prompt)

        # Generate with reward guidance
        image = sampler.sample_reward_guided(
            reward_type="ensemble",
            reward_model=reward_ensemble,
            num_steps=steps,
            guidance_scale=guidance_scale,
            step_size=args.step_size,
            img_shape=(resolution, resolution),
            seed=args.seed,
            num_optim_iters=args.num_optim_iters,
            grad_mode=args.grad_mode,
            sample_mode=sample_mode,
            noise_mode='deterministic',
            grad_norm_mode=grad_norm_mode_internal,
            early_stop=early_stop,
            use_adam=True,
            enable_callback=False,
            grad_checkpointing=args.grad_checkpointing,
            warmup_steps=args.warmup_steps,
            warmup_particles=args.warmup_particles,
            warmup_lr=args.warmup_lr,
            unguided_steps=args.unguided_steps,
        )

        # Save in GenEval format
        os.makedirs(os.path.join(outpath, "samples"), exist_ok=True)
        with open(os.path.join(outpath, "metadata.jsonl"), "w") as f:
            json.dump(metadata, f)

        # Convert tensor from [-1,1] to [0,1] and save
        from torchvision.utils import save_image
        save_image(((image + 1) / 2).clamp(0, 1), img_path)

        completed += 1
        elapsed = time.time() - t_start
        rate = elapsed / completed if completed > 0 else 0
        remaining = rate * (total - skipped - completed)
        print(f"[{idx}/{end_idx}] {prompt[:60]}... ({rate:.1f}s/img, ~{remaining/60:.0f}m left)")

    elapsed = time.time() - t_start
    print(f"\nDone: {completed} generated, {skipped} skipped, {elapsed/60:.1f}m total")


if __name__ == "__main__":
    main()
