#!/bin/bash
set -euo pipefail

CKPT_ROOT="${FMRG_CKPT_ROOT:-$(pwd)/checkpoints}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$CKPT_ROOT" "$HF_HOME"

HF_REPO="gabeguofanclub/flux-1-dev-flowmap-lsd"
LORA_256_PATH="01-05-26/runs/uniform_half_plane_lr_3e-4/checkpoint-8500/pytorch_lora_weights.safetensors"
LORA_512_PATH="01-12-26/runs/res_512_steps_50k_rank_64_lr_1e-4/checkpoint-43000/pytorch_lora_weights.safetensors"
LORA_DIR_256="$CKPT_ROOT/flux-flowmap-lora"
LORA_DIR_512="$CKPT_ROOT/flux-flowmap-lora-512"

if [ ! -f "$LORA_DIR_256/pytorch_lora_weights.safetensors" ]; then
    mkdir -p "$LORA_DIR_256"
    huggingface-cli download "$HF_REPO" "$LORA_256_PATH" --local-dir "$LORA_DIR_256" --local-dir-use-symlinks False
    mv "$LORA_DIR_256/$LORA_256_PATH" "$LORA_DIR_256/pytorch_lora_weights.safetensors"
    rm -rf "$LORA_DIR_256/01-05-26"
fi

if [ ! -f "$LORA_DIR_512/pytorch_lora_weights.safetensors" ]; then
    mkdir -p "$LORA_DIR_512"
    huggingface-cli download "$HF_REPO" "$LORA_512_PATH" --local-dir "$LORA_DIR_512" --local-dir-use-symlinks False
    mv "$LORA_DIR_512/$LORA_512_PATH" "$LORA_DIR_512/pytorch_lora_weights.safetensors"
    rm -rf "$LORA_DIR_512/01-12-26"
fi

# FLUX.1-Dev is gated. Accept the license at
# https://huggingface.co/black-forest-labs/FLUX.1-dev and run `huggingface-cli login`.
huggingface-cli download black-forest-labs/FLUX.1-dev --quiet || true

# Pre-warm the ReNO reward caches (HPSv2 / ImageReward / PickScore / CLIP).
python - <<'PY' || true
try: import hpsv2
except Exception: pass
try: import ImageReward
except Exception: pass
try:
    from huggingface_hub import snapshot_download
    snapshot_download("yuvalkirstain/PickScore_v1")
    snapshot_download("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
except Exception: pass
PY
