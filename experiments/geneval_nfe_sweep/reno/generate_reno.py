"""ReNO optimization with FLUX FlowMap."""

import json
import logging
import math
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
sys.path.insert(0, REPO_ROOT)

import torch
from pytorch_lightning import seed_everything
from tqdm import tqdm

_reno_repo = os.environ.get("RENO_REPO", os.path.join(SCRIPT_DIR, "ReNO"))
if os.path.exists(_reno_repo):
    sys.path.insert(0, _reno_repo)

from rewards import get_reward_losses
from training import LatentNoiseTrainer, get_optimizer

from flux_flowmap_model import RewardFluxFlowMapPipeline


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="ReNO with FLUX FlowMap")

    parser.add_argument("--cache_dir", type=str,
                        default=os.environ.get("HF_HOME"),
                        help="HF cache directory")
    parser.add_argument("--save_dir", type=str, default=os.path.join(SCRIPT_DIR, "outputs"),
                        help="Directory to save images")

    # FLUX FlowMap specific
    parser.add_argument("--resolution", type=int, default=512, choices=[256, 512],
                        help="Image resolution")
    parser.add_argument("--guidance_scale", type=float, default=3.5,
                        help="FLUX guidance scale")
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA weights (auto-selected by resolution if None)")

    # Optimization
    parser.add_argument("--lr", type=float, default=5.0, help="Learning rate")
    parser.add_argument("--n_iters", type=int, default=50, help="Number of iterations")
    parser.add_argument("--n_inference_steps", type=int, default=1,
                        help="Number of inference steps (1 for FlowMap one-step)")
    parser.add_argument("--multi_step", type=int, default=8,
                        help="Number of flow map steps for final image (0 to disable)")
    parser.add_argument("--optim", choices=["sgd", "adam", "lbfgs"], default="sgd")
    parser.add_argument("--nesterov", default=True, action="store_false")
    parser.add_argument("--grad_clip", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)

    # Reward losses
    parser.add_argument("--disable_hps", default=True, action="store_false", dest="enable_hps")
    parser.add_argument("--hps_weighting", type=float, default=5.0)
    parser.add_argument("--disable_imagereward", default=True, action="store_false", dest="enable_imagereward")
    parser.add_argument("--imagereward_weighting", type=float, default=1.0)
    parser.add_argument("--disable_clip", default=True, action="store_false", dest="enable_clip")
    parser.add_argument("--clip_weighting", type=float, default=0.01)
    parser.add_argument("--disable_pickscore", default=True, action="store_false", dest="enable_pickscore")
    parser.add_argument("--pickscore_weighting", type=float, default=0.05)
    parser.add_argument("--disable_aesthetic", default=False, action="store_false", dest="enable_aesthetic")
    parser.add_argument("--aesthetic_weighting", type=float, default=0.0)
    parser.add_argument("--disable_reg", default=True, action="store_false", dest="enable_reg")
    parser.add_argument("--reg_weight", type=float, default=0.01)

    # Task
    parser.add_argument("--task", type=str, default="single",
                        choices=["t2i-compbench", "single", "parti-prompts", "geneval", "example-prompts"])
    parser.add_argument("--prompt", type=str, default="A red dog and a green cat")
    parser.add_argument("--benchmark_reward", default="total",
                        choices=["ImageReward", "PickScore", "HPS", "CLIP", "total"])

    # General
    parser.add_argument("--save_all_images", default=False, action="store_true")
    parser.add_argument("--no_optim", default=False, action="store_true")
    parser.add_argument("--imageselect", default=False, action="store_true")
    parser.add_argument("--memsave", default=False, action="store_true")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--device_id", type=str, default=None)

    # Multi-step (not used for FlowMap, kept for compatibility)
    parser.add_argument("--enable_multi_apply", default=False, action="store_true")
    parser.add_argument("--multi_step_model", type=str, default="flux")

    # Sharding (for parallel SLURM jobs)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Start prompt index (inclusive)")
    parser.add_argument("--end_idx", type=int, default=-1,
                        help="End prompt index (exclusive), -1 for all")

    return parser.parse_args()


def collect_prompts(args):
    """Collect all prompts for a task so we can pre-encode them."""
    prompts = []
    if args.task == "single":
        prompts = [args.prompt]
    elif args.task == "example-prompts":
        with open(os.path.join(SCRIPT_DIR, "ReNO/assets/example_prompts.txt"), "r") as f:
            prompts = [p.strip() for p in f.readlines()]
    elif args.task == "t2i-compbench":
        prompt_file = os.path.join(SCRIPT_DIR, "T2I-CompBench/examples/dataset", f"{args.prompt}.txt")
        with open(prompt_file, "r") as f:
            prompts = [p.strip() for p in f.readlines()]
    elif args.task == "parti-prompts":
        from datasets import load_dataset
        parti_dataset = load_dataset("nateraw/parti-prompts", split="train")
        prompts = [s["Prompt"] for s in parti_dataset]
    elif args.task == "geneval":
        prompt_file = os.path.join(SCRIPT_DIR, "geneval/prompts/evaluation_metadata.jsonl")
        with open(prompt_file) as fp:
            metadatas = [json.loads(line) for line in fp]
        prompts = [m["prompt"] for m in metadatas]
    return prompts


def main(args):
    seed_everything(args.seed)
    os.makedirs(f"{args.save_dir}/logs/{args.task}", exist_ok=True)

    # Logging
    logger = logging.getLogger()
    settings = (
        f"fluxfm_{args.resolution}"
        f"{'_' + args.prompt if args.task == 't2i-compbench' else ''}"
        f"{'_no-optim' if args.no_optim else ''}_{args.seed if args.task != 'geneval' else ''}"
        f"_lr{args.lr}_gc{args.grad_clip}_iter{args.n_iters}"
        f"_reg{args.reg_weight if args.enable_reg else '0'}"
        f"{'_pickscore' + str(args.pickscore_weighting) if args.enable_pickscore else ''}"
        f"{'_clip' + str(args.clip_weighting) if args.enable_clip else ''}"
        f"{'_hps' + str(args.hps_weighting) if args.enable_hps else ''}"
        f"{'_imagereward' + str(args.imagereward_weighting) if args.enable_imagereward else ''}"
        f"{'_aesthetic' + str(args.aesthetic_weighting) if args.enable_aesthetic else ''}"
    )
    file_stream = open(f"{args.save_dir}/logs/{args.task}/{settings}.txt", "w")
    handler = logging.StreamHandler(file_stream)
    formatter = logging.Formatter("%(asctime)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel("INFO")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)
    logging.info(args)

    if args.device_id is not None:
        logging.info(f"Using CUDA device {args.device_id}")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    # Step 1: Load FLUX FlowMap model (text encoders still loaded)
    pipe = RewardFluxFlowMapPipeline(
        resolution=args.resolution,
        device=device,
        dtype=torch.bfloat16,  # FLUX always uses bfloat16
        guidance_scale=args.guidance_scale,
        lora_path=args.lora_path,
    )

    # Step 2: Pre-encode all prompts for this task
    prompts = collect_prompts(args)
    pipe.precompute_embeddings(prompts)

    # Step 3: Load reward models (text encoders are now deleted, saving ~10GB VRAM)
    reward_losses = get_reward_losses(args, dtype, device, args.cache_dir)

    # Step 4: Create multi_apply_fn for multi-step final generation
    multi_apply_fn = None
    if args.multi_step > 0:
        multi_apply_fn = lambda latents, prompt: torch.no_grad()(
            lambda l, p: pipe.multi_apply(l, p, num_inference_steps=args.multi_step)
        )(latents, prompt)
        logging.info(f"Multi-step final generation enabled: {args.multi_step} flow map steps")

    # Step 5: Create trainer
    trainer = LatentNoiseTrainer(
        reward_losses=reward_losses,
        model=pipe,
        n_iters=args.n_iters,
        n_inference_steps=args.n_inference_steps,
        seed=args.seed,
        save_all_images=args.save_all_images,
        device=device,
        no_optim=args.no_optim,
        regularize=args.enable_reg,
        regularization_weight=args.reg_weight,
        grad_clip=args.grad_clip,
        log_metrics=args.task == "single" or not args.no_optim,
        imageselect=args.imageselect,
    )

    # Latent shape (packed form for FLUX FlowMap)
    shape = pipe.latent_shape
    enable_grad = not args.no_optim

    if args.task == "single":
        init_latents = torch.randn(shape, device=device, dtype=torch.bfloat16)
        latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
        optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
        save_dir = f"{args.save_dir}/{args.task}/{settings}/{args.prompt[:150]}"
        os.makedirs(save_dir, exist_ok=True)
        init_image, best_image, total_init_rewards, total_best_rewards = trainer.train(
            latents, args.prompt, optimizer, save_dir, multi_apply_fn=multi_apply_fn,
        )
        best_image.save(f"{save_dir}/best_image.png")
        init_image.save(f"{save_dir}/init_image.png")

    elif args.task == "example-prompts":
        with open(os.path.join(SCRIPT_DIR, "ReNO/assets/example_prompts.txt"), "r") as fo:
            prompts = [p.strip() for p in fo.readlines()]
        for i, prompt in tqdm(enumerate(prompts)):
            init_latents = torch.randn(shape, device=device, dtype=torch.bfloat16)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
            name = f"{i:03d}_{prompt[:150]}.png"
            save_dir = f"{args.save_dir}/{args.task}/{settings}/{name}"
            os.makedirs(save_dir, exist_ok=True)
            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer, save_dir
            )
            if i == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in best_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
            best_image.save(f"{save_dir}/best_image.png")
            init_image.save(f"{save_dir}/init_image.png")
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= len(prompts)
            total_init_rewards[k] /= len(prompts)
        os.makedirs(f"{args.save_dir}/example-prompts/{settings}", exist_ok=True)
        with open(f"{args.save_dir}/example-prompts/{settings}/results.txt", "w") as f:
            f.write(f"Mean initial all rewards: {total_init_rewards}\n"
                    f"Mean best all rewards: {total_best_rewards}\n")

    elif args.task == "t2i-compbench":
        prompt_file = os.path.join(SCRIPT_DIR, "T2I-CompBench/examples/dataset", f"{args.prompt}.txt")
        with open(prompt_file, "r") as fo:
            prompts = [p.strip() for p in fo.readlines()]
        os.makedirs(f"{args.save_dir}/{args.task}/{settings}/samples", exist_ok=True)
        for i, prompt in tqdm(enumerate(prompts)):
            init_latents = torch.randn(shape, device=device, dtype=torch.bfloat16)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer, None
            )
            if i == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in best_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
            name = f"{prompt}_{i:06d}.png"
            best_image.save(f"{args.save_dir}/{args.task}/{settings}/samples/{name}")
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= len(prompts)
            total_init_rewards[k] /= len(prompts)

    elif args.task == "parti-prompts":
        from datasets import load_dataset
        parti_dataset = load_dataset("nateraw/parti-prompts", split="train")
        total_reward_diff = 0.0
        total_best_reward = 0.0
        total_init_reward = 0.0
        total_improved_samples = 0
        for index, sample in enumerate(parti_dataset):
            init_latents = torch.randn(shape, device=device, dtype=torch.bfloat16)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
            os.makedirs(f"{args.save_dir}/{args.task}/{settings}/{index}", exist_ok=True)
            prompt = sample["Prompt"]
            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer
            )
            best_image.save(f"{args.save_dir}/{args.task}/{settings}/{index}/best_image.png")
            with open(f"{args.save_dir}/{args.task}/{settings}/{index}/prompt.txt", "w") as f:
                f.write(f"{prompt}\nInitial Rewards: {init_rewards}\nBest Rewards: {best_rewards}")
            logging.info(f"Initial rewards: {init_rewards}")
            logging.info(f"Best rewards: {best_rewards}")
            initial_reward = init_rewards[args.benchmark_reward]
            best_reward = best_rewards[args.benchmark_reward]
            total_reward_diff += best_reward - initial_reward
            total_best_reward += best_reward
            total_init_reward += initial_reward
            if best_reward < initial_reward:
                total_improved_samples += 1
            if index == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in init_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
        n_samples = parti_dataset.num_rows
        improvement_percentage = total_improved_samples / n_samples
        mean_best_reward = total_best_reward / n_samples
        mean_init_reward = total_init_reward / n_samples
        mean_reward_diff = total_reward_diff / n_samples
        logging.info(f"Improvement percentage: {improvement_percentage:.4f}, "
                     f"mean initial reward: {mean_init_reward:.4f}, "
                     f"mean best reward: {mean_best_reward:.4f}, "
                     f"mean reward diff: {mean_reward_diff:.4f}")
        for k in total_best_rewards.keys():
            total_best_rewards[k] /= n_samples
            total_init_rewards[k] /= n_samples
        os.makedirs(f"{args.save_dir}/parti-prompts/{settings}", exist_ok=True)
        with open(f"{args.save_dir}/parti-prompts/{settings}/results.txt", "w") as f:
            f.write(f"Mean improvement: {improvement_percentage:.4f}, "
                    f"mean initial reward: {mean_init_reward:.4f}, "
                    f"mean best reward: {mean_best_reward:.4f}, "
                    f"mean reward diff: {mean_reward_diff:.4f}\n"
                    f"Mean initial all rewards: {total_init_rewards}\n"
                    f"Mean best all rewards: {total_best_rewards}")

    elif args.task == "geneval":
        prompt_file = os.path.join(SCRIPT_DIR, "geneval/prompts/evaluation_metadata.jsonl")
        with open(prompt_file) as fp:
            metadatas = [json.loads(line) for line in fp]
        # Sharding support
        start_idx = args.start_idx
        end_idx = args.end_idx if args.end_idx > 0 else len(metadatas)
        end_idx = min(end_idx, len(metadatas))
        logging.info(f"GenEval: processing prompts {start_idx} to {end_idx} (of {len(metadatas)})")
        outdir = f"{args.save_dir}/{args.task}/{settings}"
        count = 0
        for index in range(start_idx, end_idx):
            metadata = metadatas[index]
            outpath = f"{outdir}/{index:0>5}"
            # Resume: skip if image already exists
            img_path = f"{outpath}/samples/{args.seed:05}.png"
            if os.path.exists(img_path):
                logging.info(f"Skipping {index} (already exists)")
                continue
            init_latents = torch.randn(shape, device=device, dtype=torch.bfloat16)
            latents = torch.nn.Parameter(init_latents, requires_grad=enable_grad)
            optimizer = get_optimizer(args.optim, latents, args.lr, args.nesterov)
            prompt = metadata["prompt"]
            init_image, best_image, init_rewards, best_rewards = trainer.train(
                latents, prompt, optimizer, None, multi_apply_fn=multi_apply_fn,
            )
            logging.info(f"[{index}/{end_idx}] Initial rewards: {init_rewards}")
            logging.info(f"[{index}/{end_idx}] Best rewards: {best_rewards}")
            os.makedirs(f"{outpath}/samples", exist_ok=True)
            with open(f"{outpath}/metadata.jsonl", "w") as fp:
                json.dump(metadata, fp)
            best_image.save(img_path)
            if count == 0:
                total_best_rewards = {k: 0.0 for k in best_rewards.keys()}
                total_init_rewards = {k: 0.0 for k in best_rewards.keys()}
            for k in best_rewards.keys():
                total_best_rewards[k] += best_rewards[k]
                total_init_rewards[k] += init_rewards[k]
            count += 1
        if count > 0:
            for k in total_best_rewards.keys():
                total_best_rewards[k] /= count
                total_init_rewards[k] /= count
        else:
            total_best_rewards = {}
            total_init_rewards = {}
    else:
        raise ValueError(f"Unknown task {args.task}")

    logging.info(f"Mean initial rewards: {total_init_rewards}")
    logging.info(f"Mean best rewards: {total_best_rewards}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
