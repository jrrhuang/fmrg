"""
Generate artistic images: unguided (base FLUX flow map) and FMRG-J guided.

Usage:
    python scripts/generate_aesthetic.py --mode unguided --output_dir icml/artistic_outputs/unguided
    python scripts/generate_aesthetic.py --mode guided --output_dir icml/artistic_outputs/guided
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

# Must be imported before fluxfm_sampler_reward (which lazy-imports ImageReward)
# to monkey-patch transformers 5.x APIs that ImageReward expects from 4.x.
import imagereward_compat  # noqa: F401

from fluxfm_sampler_reward import FluxFlowMapSampler, RewardEnsemble

LORA_PATH = os.environ.get("FMRG_LORA_PATH", os.path.join(REPO_ROOT, "checkpoints", "flux-flowmap-lora-512"))


def load_prompts(path):
    prompts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    return prompts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts_file", type=str,
                        default=os.path.join(REPO_ROOT, "data", "artistic_prompts.txt"))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["unguided", "guided"], required=True,
                        help="unguided = plain flow-map sampling. guided = FMRG-J with reward ensemble.")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0,
                        help="First seed; sample i uses seed + i.")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Number of consecutive seeds to generate per prompt.")
    parser.add_argument("--unguided_steps_fm", type=int, default=8,
                        help="Flow-map steps when --mode unguided.")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=0, help="0 = all")
    # Guided FMRG-J config.
    parser.add_argument("--num_steps", type=int, default=32, help="Timestep schedule resolution for guided mode.")
    parser.add_argument("--early_stop", type=int, default=8,
                        help="Early stop at step N; 0=disabled.")
    parser.add_argument("--warmup_steps", type=int, default=4,
                        help="Reinitialization steps.")
    parser.add_argument("--warmup_particles", type=int, default=3,
                        help="Particles per reinitialization.")
    parser.add_argument("--step_size", type=float, default=3.0,
                        help="Guidance strength λ.")
    parser.add_argument("--unguided_steps", type=int, default=2,
                        help="Trailing uncontrolled flow-map steps to t=0.")
    parser.add_argument("--sample_mode", type=str, default="flow_map1", choices=["flow_map1", "flow_map2"],
                        help="flow_map1 = 1-NFE/step. flow_map2 = full 2-NFE/step.")
    parser.add_argument("--save_intermediates", action="store_true",
                        help="Save predicted-x0 image at every guided step (writes a per-seed lookahead/ subdir)")
    parser.add_argument("--progress_lookahead_steps", type=int, default=0,
                        help="If > 0, progress images are computed by running this many flow-matching steps "
                             "from the current z_t down to t=0 (used only for saved progress images, "
                             "the main sampling algorithm is unchanged)")
    parser.add_argument("--grad_checkpointing", type=lambda s: s.lower() == 'true', default=True,
                        help="Enable gradient checkpointing in transformer (saves VRAM, ~2x slower backward)")
    parser.add_argument("--force_particle_idx", type=int, default=None,
                        help="If set, override the warmup reward-selection and force this particle index.")
    args = parser.parse_args()

    device = "cuda"
    guidance_scale = 3.5

    prompts = load_prompts(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts")

    print("Initializing FluxFlowMapSampler...")
    sampler = FluxFlowMapSampler(
        prompt=prompts[0],
        additional_prompts=prompts[1:] if len(prompts) > 1 else None,
        lora_path=LORA_PATH,
        device=device,
    )

    # Load reward ensemble for guided mode
    reward_ensemble = None
    if args.mode == "guided":
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        print("Loading reward ensemble...")
        reward_ensemble = RewardEnsemble(
            device=device, dtype=torch.float32, cache_dir=cache_dir,
            hps_weight=5.0, imagereward_weight=1.0,
            pickscore_weight=0.05, clip_weight=0.01,
        )

    os.makedirs(args.output_dir, exist_ok=True)

    end_idx = args.end_idx if args.end_idx > 0 else len(prompts)
    prompts = prompts[args.start_idx:end_idx]

    for pi, prompt in enumerate(prompts):
        print(f"\n[{pi+1}/{len(prompts)}] {prompt[:60]}...")
        sampler.set_prompt_embeddings(prompt)

        abs_pi = pi + args.start_idx
        prompt_dir = os.path.join(args.output_dir, f"{abs_pi:03d}")
        os.makedirs(prompt_dir, exist_ok=True)

        # Save prompt
        with open(os.path.join(prompt_dir, "prompt.txt"), "w") as f:
            f.write(prompt)

        for seed in range(args.seed, args.seed + args.num_seeds):
            img_path = os.path.join(prompt_dir, f"seed_{seed:04d}.png")
            if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                print(f"  seed {seed}: exists, skipping")
                continue

            if args.mode == "unguided":
                result = sampler.sample_forward(
                    num_steps=args.unguided_steps_fm,
                    height=args.resolution,
                    width=args.resolution,
                    guidance_scale=guidance_scale,
                    seed=seed,
                )
                pil_img = result[0]
                pil_img.save(img_path)
            else:
                # Optional callback to save lookahead intermediates (predicted-x0 per step).
                cb = None
                la_dir = None
                if args.save_intermediates:
                    la_dir = os.path.join(prompt_dir, f"lookahead_seed_{seed:04d}")
                    os.makedirs(la_dir, exist_ok=True)
                    def cb(step_idx, img_tensor, t_next, reward_loss, _la_dir=la_dir):
                        # img_tensor in [-1, 1]; convert and save with frame idx (1-based, frame 0 = unguided)
                        out = os.path.join(_la_dir, f"frame_{step_idx:03d}_t{t_next:.3f}_loss{float(reward_loss):.3f}.png")
                        save_image(((img_tensor + 1) / 2).clamp(0, 1), out)
                image = sampler.sample_reward_guided(
                    reward_model=reward_ensemble,
                    num_steps=args.num_steps,
                    guidance_scale=guidance_scale,
                    step_size=args.step_size,
                    img_shape=(args.resolution, args.resolution),
                    seed=seed,
                    num_optim_iters=1,
                    # FMRG-J (Jacobian) with grad rescaled to velocity norm
                    grad_mode="jac",
                    sample_mode=args.sample_mode,
                    noise_mode="deterministic",
                    grad_norm_mode="normalize",
                    early_stop=args.early_stop,
                    use_adam=True,
                    enable_callback=bool(args.save_intermediates),
                    callback=cb,
                    grad_checkpointing=args.grad_checkpointing,
                    warmup_steps=args.warmup_steps,
                    warmup_particles=args.warmup_particles,
                    unguided_steps=args.unguided_steps,
                    progress_lookahead_steps=args.progress_lookahead_steps,
                    force_particle_idx=args.force_particle_idx,
                )
                save_image(((image + 1) / 2).clamp(0, 1), img_path)

                # Save the actual final guided image as the last frame in the lookahead dir
                if args.save_intermediates and la_dir is not None:
                    existing = sorted([f for f in os.listdir(la_dir) if f.startswith("frame_")])
                    last_idx = max([int(f.split("_")[1]) for f in existing], default=0)
                    final_frame_path = os.path.join(la_dir, f"frame_{last_idx + 1:03d}_final.png")
                    save_image(((image + 1) / 2).clamp(0, 1), final_frame_path)

                # Save unguided version using the selected particle's seed
                selected_p = getattr(sampler, '_selected_particle_idx', 0)
                actual_seed = seed + selected_p
                unguided_path = os.path.join(prompt_dir, f"unguided_seed_{seed:04d}_p{selected_p}.png")
                if args.save_intermediates and la_dir is not None:
                    # Generate matching unguided image and save as frame 0 inside lookahead dir.
                    frame0_path = os.path.join(la_dir, "frame_000_unguided.png")
                    if not os.path.exists(frame0_path):
                        result = sampler.sample_forward(
                            num_steps=8,
                            height=args.resolution,
                            width=args.resolution,
                            guidance_scale=guidance_scale,
                            seed=actual_seed,
                        )
                        result[0].save(frame0_path)
                if not os.path.exists(unguided_path):
                    result = sampler.sample_forward(
                        num_steps=8,
                        height=args.resolution,
                        width=args.resolution,
                        guidance_scale=guidance_scale,
                        seed=actual_seed,
                    )
                    result[0].save(unguided_path)
                    print(f"  seed {seed}: guided saved (particle {selected_p}), unguided saved (seed {actual_seed})")
                else:
                    print(f"  seed {seed}: guided saved (particle {selected_p}), unguided exists")

                # Save particle info
                import json
                info_path = os.path.join(prompt_dir, f"seed_{seed:04d}_info.json")
                with open(info_path, "w") as f:
                    json.dump({"seed": seed, "selected_particle": selected_p, "actual_seed": actual_seed}, f)


if __name__ == "__main__":
    main()
