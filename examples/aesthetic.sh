#!/bin/bash
# Reward-guided aesthetic generation (FMRG-J).

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
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/aesthetic}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-1}"

python scripts/generate_aesthetic.py \
    --mode guided \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --resolution 512 \
    --seeds 0 \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --num_steps 16 \
    --early_stop 4 \
    --warmup_steps 0 \
    --warmup_particles 1 \
    --step_size 3.0 \
    --unguided_steps 2 \
    --sample_mode flow_map1 \
    --grad_checkpointing true

echo "Output in $OUTPUT_DIR/"
