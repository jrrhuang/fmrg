#!/bin/bash
set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../../.." && pwd )"
cd "$REPO_ROOT"

if [[ ! -d "$SCRIPT_DIR/ReNO" ]]; then
    git clone --depth 1 https://github.com/ExplainableML/ReNO.git "$SCRIPT_DIR/ReNO"
fi
export RENO_REPO="$SCRIPT_DIR/ReNO"

mkdir -p "$SCRIPT_DIR/geneval"
ln -sfn "$REPO_ROOT/data/geneval_prompts" "$SCRIPT_DIR/geneval/prompts"

OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/geneval_nfe_sweep/reno}"
SEEDS="${SEEDS:-0 1 2 3}"

# tag:n_iters:multi_step:lr
CONFIGS=(
    "reno_nfe9:5:4:5.0"
    "reno_nfe18:10:8:5.0"
    "reno_nfe33:25:8:5.0"
    "reno_nfe58:50:8:3.0"
    "reno_nfe108:100:8:1.0"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r tag n_iters multi_step lr <<< "$cfg"
    for SEED in $SEEDS; do
        SAVE_DIR="$OUTPUT_ROOT/${tag}_seed${SEED}"
        if [[ -f "$SAVE_DIR/results.jsonl" ]]; then
            echo "SKIP $tag seed=$SEED"
            continue
        fi
        echo ">> $tag seed=$SEED"
        python "$SCRIPT_DIR/generate_reno.py" \
            --task geneval \
            --save_dir "$SAVE_DIR" \
            --resolution 512 \
            --guidance_scale 3.5 \
            --n_iters "$n_iters" \
            --multi_step "$multi_step" \
            --lr "$lr" \
            --seed "$SEED" \
            --reward_hps_weighting 5.0 \
            --reward_imagereward_weighting 1.0 \
            --reward_pickscore_weighting 0.05 \
            --reward_clip_weighting 0.01
    done
done
