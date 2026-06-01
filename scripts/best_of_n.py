"""
Best-of-N baseline: unguided sampling + reward rerank.

For each prompt, sample N images unguided (one per seed), score them with the
reward ensemble, and keep the highest-scoring image.

Usage:
    python scripts/best_of_n.py \\
        --prompts_file data/artistic_prompts.txt \\
        --output_dir ./results/best_of_n \\
        --n 8 --resolution 512 --seed_base 0
"""

import argparse
import json
import os
import sys

import torch
from torchvision.utils import save_image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)

# Must precede fluxfm_sampler_reward (which lazy-imports ImageReward) under
# transformers 5.x.
import imagereward_compat  # noqa: F401

from fluxfm_sampler_reward import FluxFlowMapSampler, RewardEnsemble


LORA_PATH = os.environ.get(
    "FMRG_LORA_PATH", os.path.join(REPO_ROOT, "checkpoints", "flux-flowmap-lora-512")
)


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Best-of-N unguided generation baseline")
    parser.add_argument("--prompts_file", type=str, required=True,
                        help="Text file with one prompt per line (# comments skipped)")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n", type=int, default=8,
                        help="Number of samples per prompt")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num_steps", type=int, default=8,
                        help="Unguided flow-map steps per sample")
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed_base", type=int, default=0,
                        help="First seed; sample i uses seed_base + i")
    parser.add_argument("--save_all", action="store_true",
                        help="Also save every candidate (not just the winner)")

    # Reward ensemble weights
    parser.add_argument("--hps_weight", type=float, default=5.0)
    parser.add_argument("--imagereward_weight", type=float, default=1.0)
    parser.add_argument("--pickscore_weight", type=float, default=0.05)
    parser.add_argument("--clip_weight", type=float, default=0.01)

    parser.add_argument("--cache_dir", type=str,
                        default=os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")))
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0, help="0 = all")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompts = load_prompts(args.prompts_file)
    end_idx = args.end_idx if args.end_idx > 0 else len(prompts)
    prompts = prompts[args.start_idx:end_idx]
    print(f"Loaded {len(prompts)} prompts")

    print("Initializing FluxFlowMapSampler...")
    sampler = FluxFlowMapSampler(
        prompt=prompts[0],
        additional_prompts=prompts[1:] if len(prompts) > 1 else None,
        lora_path=LORA_PATH,
        device=device,
    )

    print("Loading reward ensemble...")
    reward_ensemble = RewardEnsemble(
        device=device, dtype=torch.float32, cache_dir=args.cache_dir,
        hps_weight=args.hps_weight, imagereward_weight=args.imagereward_weight,
        pickscore_weight=args.pickscore_weight, clip_weight=args.clip_weight,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    for pi, prompt in enumerate(prompts):
        abs_pi = pi + args.start_idx
        prompt_dir = os.path.join(args.output_dir, f"{abs_pi:03d}")
        os.makedirs(prompt_dir, exist_ok=True)
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        best_path = os.path.join(prompt_dir, "best.png")
        if os.path.exists(best_path) and os.path.getsize(best_path) > 0:
            print(f"[{pi+1}/{len(prompts)}] {prompt[:60]}... best exists, skipping")
            continue

        print(f"\n[{pi+1}/{len(prompts)}] {prompt[:60]}...")
        sampler.set_prompt_embeddings(prompt)

        # Generate N candidates
        candidates = []  # list of (seed, image_tensor_neg1_to_1)
        for i in range(args.n):
            seed = args.seed_base + i
            with torch.no_grad():
                pil = sampler.sample_forward(
                    num_steps=args.num_steps,
                    height=args.resolution,
                    width=args.resolution,
                    guidance_scale=args.guidance_scale,
                    seed=seed,
                )[0]
            img_tensor = torch.from_numpy(
                __import__('numpy').array(pil)
            ).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
            img_tensor = img_tensor.to(device)
            candidates.append((seed, img_tensor, pil))

        # Score each candidate. RewardEnsemble.__call__ returns
        # (total_loss, losses_dict); lower loss = higher reward.
        scores = []
        for seed, img_tensor, pil in candidates:
            with torch.no_grad():
                img_01 = ((img_tensor + 1) / 2).clamp(0, 1)
                total_loss, losses_dict = reward_ensemble(img_01, prompt)
                reward = -float(total_loss.item())
            scores.append({"seed": seed, "reward": reward, "losses": losses_dict})

        # Pick best (highest reward)
        best_idx = max(range(args.n), key=lambda i: scores[i]["reward"])
        best_seed, best_tensor, best_pil = candidates[best_idx]

        # Save winner
        save_image(((best_tensor + 1) / 2).clamp(0, 1), best_path)

        # Save metadata
        with open(os.path.join(prompt_dir, "best_of_n_info.json"), "w") as f:
            json.dump({"selected_seed": best_seed, "n": args.n, "scores": scores}, f, indent=2)

        # Optionally save every candidate
        if args.save_all:
            for (seed, t, _), s in zip(candidates, scores):
                save_image(((t + 1) / 2).clamp(0, 1),
                           os.path.join(prompt_dir, f"seed_{seed:04d}_r{s['reward']:.3f}.png"))

        print(f"  selected seed {best_seed} (reward {scores[best_idx]['reward']:.3f}) "
              f"out of {args.n} candidates")


if __name__ == "__main__":
    main()
