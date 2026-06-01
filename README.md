# FMRG — Flow Map Reward Guidance

[![ICML 2026](https://img.shields.io/badge/ICML-2026-blue.svg)](https://icml.cc/)
[![arXiv](https://img.shields.io/badge/arXiv-2604.27147-b31b1b.svg)](https://arxiv.org/abs/2604.27147)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](LICENSE)

https://arxiv.org/abs/2604.27147

by Jerry Y. Huang, Justin Lin, Sheel Shah, Kartik Nair, and Nicholas M. Boffi.

FMRG is a training-free framework for inference-time alignment of pre-trained
flow maps. Guidance is cast as a deterministic optimal-control problem over the
flow map's trajectory; two practical algorithms — Jacobian (**FMRG-J**) and
Euclidean (**FMRG-E**) — fall out of the first-order conditions. The same
framework covers classical latent-space inverse problems and learned reward
objectives in only a few NFEs per sample. The release also ships FLUX-FlowMap
ports of the FlowDPS and FlowChef baselines (`--method {flowdps, flowchef}`).

## Installation

### Requirements

- Python 3.11
- CUDA-capable GPU

### Setup

```bash
git clone https://github.com/jrrhuang/fmrg.git
cd fmrg
bash install.sh
conda activate fmrg_env
bash checkpoints_download.sh
```

`install.sh` creates a Python 3.11 / torch 2.5.1+cu121 environment from
`environment.yml` and pip-installs `clip`, `image-reward`, and `hpsv2` with
`--no-build-isolation` (their `setup.py` modules require legacy
`pkg_resources`).

`checkpoints_download.sh` fetches two FLUX FlowMap LoRAs into `checkpoints/`:

- `flux-flowmap-lora/` (256-res) — used by inverse problems.
- `flux-flowmap-lora-512/` (512-res) — used by reward-guided generation.

It also pre-warms the reward-model caches under `$HF_HOME`.

## Quick start

### Inverse problems

```bash
python scripts/solve_inverse_problem.py \
    --task_config configs/inverse_problems/sr_config.yaml \
    --image_path /path/to/image.png \
    --method fmrg --grad_mode jac --normalize_grad \
    --num_steps 30 --step_size 5.0 --num_optim_iters 1 \
    --sample_mode flow_map1 --loss_mode pixel \
    --prompt "a photo of a dog" --seed 0 \
    --save_dir ./results/sr --resolution 256
```

Available task configs:

| Config | Task |
|---|---|
| `configs/inverse_problems/sr_config.yaml` | 4× super-resolution |
| `configs/inverse_problems/motion_deblur_config.yaml` | 61×61 motion deblur |
| `configs/inverse_problems/box_inpainting_64_config.yaml` | centered 64×64 box inpainting |

`--resolution {256, 512}` auto-selects the matching LoRA. `--normalize_grad`
rescales each per-iteration gradient to the velocity norm.

### Reward-guided generation

```bash
python scripts/generate_aesthetic.py --mode guided \
    --prompts_file data/artistic_prompts.txt \
    --output_dir ./results/aesthetic --resolution 512 --seeds 0 \
    --start_idx 0 --end_idx 1 \
    --nfe 13 --early_stop 5 --warmup_steps 2 --warmup_particles 3 \
    --step_size 3.0 --unguided_steps 2 --sample_mode flow_map1
```

```bash
python scripts/generate_geneval.py \
    --grad_mode jac --normalize_grad --sample_mode flow_map2 \
    --nfe 20 --step_size 3.0 --num_optim_iters 1 \
    --early_stop 5 --warmup_steps 4 --warmup_particles 3 \
    --grad_checkpointing \
    --prompts_file data/geneval_prompts/evaluation_metadata.jsonl \
    --output_dir ./results/geneval --start_idx 0 --end_idx 1 --num_samples 1
```

```bash
python scripts/best_of_n.py \
    --prompts_file data/artistic_prompts.txt \
    --output_dir ./results/best_of_n \
    --n 8 --resolution 512 --num_steps 8
```

### Metrics

Dataset-level PSNR / SSIM / LPIPS / FID / KID over an inverse-problem run:

```bash
python scripts/aggregate_metrics.py --save_dir ./results/sr --gt_path /path/to/gt_dir
```

### One-command examples

`examples/` contains a shell wrapper per pipeline that calls the corresponding
script with default hyperparameters on a single input. Each runs in ~3–8 min
on one L40S.

```bash
bash examples/inverse_problems.sh
bash examples/aesthetic.sh
bash examples/geneval.sh
bash examples/best_of_n.sh
```

Override `OUTPUT_DIR`, `END_IDX`, `PROMPTS_FILE`, `IMAGE_PATH`, etc. via
environment variables.

## Project structure

```
fmrg/
├── scripts/                          # entry points
│   ├── solve_inverse_problem.py      # FMRG / FlowDPS / FlowChef on inverse problems
│   ├── generate_aesthetic.py         # FMRG-J + reward ensemble on aesthetic prompts
│   ├── generate_geneval.py           # FMRG + reward ensemble on GenEval prompts
│   ├── best_of_n.py                  # unguided + reward-ensemble rerank baseline
│   └── aggregate_metrics.py          # PSNR / SSIM / LPIPS / FID / KID aggregator
├── examples/                         # one-command shell wrappers
├── configs/inverse_problems/         # task YAMLs (SR, motion deblur, inpainting)
├── data/                             # prompts and sample inputs
├── fluxfm_sampler_ip.py              # FluxFlowMapSampler + FluxFlowDPS + FluxFlowChef
├── fluxfm_sampler_reward.py          # FluxFlowMapSampler + reward ensemble
├── functions/                        # measurement operators
├── utils/                            # image / inpaint / SVD / motion-blur helpers
├── flux_two_timestep/                # FLUX two-timestep pipeline + diffusers shim
├── checkpoints/                      # populated by checkpoints_download.sh
├── environment.yml
├── install.sh
└── checkpoints_download.sh
```

## Citation

```bibtex
@article{huang2026howtoguide,
  title={How to Guide Your Flow: Few-Step Alignment via Flow Map Reward Guidance},
  author={Huang, Jerry Y. and Lin, Justin and Shah, Sheel and Nair, Kartik and Boffi, Nicholas M.},
  journal={arXiv preprint arXiv:2604.27147},
  year={2026}
}
```

## Acknowledgments

The measurement-operator toolkit (`functions/`, `utils/`) and the FlowDPS /
FlowChef baseline classes are adapted from
[FlowDPS](https://github.com/FlowDPS-Inverse/FlowDPS). `utils/motionblur.py` is
from [LeviBorodenko/motionblur](https://github.com/LeviBorodenko/motionblur).
The FLUX FlowMap LoRAs are hosted at
[`gabeguofanclub/flux-1-dev-flowmap-lsd`](https://huggingface.co/gabeguofanclub/flux-1-dev-flowmap-lsd).

## License

Apache 2.0. See `LICENSE`.
