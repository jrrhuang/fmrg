#!/bin/bash
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/checkpoints/flux-flowmap-lora-512/pytorch_lora_weights.safetensors" ]] \
        && [[ -z "${FMRG_LORA_PATH:-}" ]]; then
    echo "ERROR: 512-res FLUX FlowMap LoRA not found. Run: bash checkpoints_download.sh"
    exit 1
fi

PROMPTS_FILE="${PROMPTS_FILE:-$REPO_ROOT/data/geneval_prompts/evaluation_metadata.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/geneval_nfe_sweep/bon}"
SEEDS="${SEEDS:-0 1 2 3}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-0}"

for N in 1 2 4 8 16 32; do
    for SEED in $SEEDS; do
        OUTPUT_DIR="$OUTPUT_ROOT/bon_N${N}_seed${SEED}"
        if [[ -f "$OUTPUT_DIR/results.jsonl" ]]; then
            echo "SKIP N=$N seed=$SEED"
            continue
        fi
        echo ">> bon_N${N} seed=$SEED"
        python scripts/best_of_n.py \
            --prompts_file "$PROMPTS_FILE" \
            --output_dir "$OUTPUT_DIR" \
            --n "$N" \
            --resolution 512 \
            --num_steps 4 \
            --seed "$SEED" \
            --start_idx "$START_IDX" \
            --end_idx "$END_IDX"
    done
done
