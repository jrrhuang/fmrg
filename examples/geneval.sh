#!/bin/bash
# Reward-guided generation on GenEval prompts (FMRG-J + ReNO ensemble).

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

PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/data/geneval_prompts/evaluation_metadata.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/geneval}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-1}"

python scripts/generate_geneval.py \
    --grad_mode jac \
    --normalize_grad \
    --sample_mode flow_map2 \
    --nfe 20 \
    --step_size 3.0 \
    --num_optim_iters 1 \
    --early_stop 5 \
    --warmup_steps 4 \
    --warmup_particles 3 \
    --grad_checkpointing \
    --resolution 512 \
    --seed 0 \
    --output_dir "$OUTPUT_DIR" \
    --prompts_file "$PROMPTS_FILE" \
    --start_idx "$START_IDX" \
    --end_idx "$END_IDX" \
    --num_samples 1

echo "Output in $OUTPUT_DIR/"
