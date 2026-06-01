#!/bin/bash
# Best-of-N: unguided sampling + reward rerank.

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/checkpoints/flux-flowmap-lora-512/pytorch_lora_weights.safetensors" ]] \
        && [[ -z "${FMRG_LORA_PATH:-}" ]]; then
    echo "ERROR: 512-res FLUX FlowMap LoRA not found."
    echo "  Run: bash checkpoints_download.sh"
    exit 1
fi

PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/data/artistic_prompts.txt}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/best_of_n}"
N="${N:-8}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-1}"

python scripts/best_of_n.py \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --n "$N" \
    --resolution 512 \
    --num_steps 8 \
    --seed 0 \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX"

echo "Output in $OUTPUT_DIR/"
