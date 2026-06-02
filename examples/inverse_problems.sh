#!/bin/bash
# Inverse problems on FLUX FlowMap: SR + motion deblur + box inpainting.

set -e

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-$0}" )" &> /dev/null && pwd )"
REPO_ROOT="$( dirname "$SCRIPT_DIR" )"
cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/checkpoints/flux-flowmap-lora/pytorch_lora_weights.safetensors" ]] \
        && [[ -z "${FMRG_LORA_PATH:-}" && -z "${FMRG_LORA_PATH_256:-}" ]]; then
    echo "ERROR: 256-res FLUX FlowMap LoRA not found."
    echo "  Run: bash checkpoints_download.sh"
    exit 1
fi

IMAGE_PATH="${IMAGE_PATH:-$REPO_ROOT/data/inverse_examples/corgi.png}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/inverse_problems}"
GT_PATH="${GT_PATH:-}"

COMMON_ARGS=(
    --image_path "$IMAGE_PATH"
    --method fmrg
    --grad_mode euc
    --num_steps 15
    --num_optim_iters 5
    --step_size 10
    --sample_mode flow_map2
    --loss_mode latent
    --seed 0
    --resolution 256
    --batch_size 1
    --prompt "a photo of a dog"
)

for task in sr motion_deblur box_inpainting_64; do
    echo "============================================================"
    echo "  $task"
    echo "============================================================"
    python scripts/solve_inverse_problem.py \
        --task_config "configs/inverse_problems/${task}_config.yaml" \
        --save_dir "$OUTPUT_ROOT/$task" \
        --compute_metrics \
        "${COMMON_ARGS[@]}"

    if [[ -n "$GT_PATH" ]]; then
        python scripts/aggregate_metrics.py \
            --save_dir "$OUTPUT_ROOT/$task" \
            --gt_path "$GT_PATH"
    fi
done

echo "Outputs in $OUTPUT_ROOT/"
