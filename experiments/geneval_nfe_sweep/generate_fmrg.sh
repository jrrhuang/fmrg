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
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/results/geneval_nfe_sweep}"
SEEDS="${SEEDS:-0 1 2 3}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:--1}"

# tag:grad_mode:sample_mode:num_steps:early_stop:warmup_steps:warmup_particles:step_size:unguided_steps:num_optim_iters
CONFIGS=(
    "fmrgj_nfe6:jac:flow_map1:16:4:0:1:2.0:2:1"
    "fmrgj_nfe11:jac:flow_map1:20:5:2:2:3.0:2:1"
    "fmrgj_nfe21:jac:flow_map1:32:8:4:3:3.0:2:1"
    "fmrgj_nfe29:jac:flow_map1:48:12:6:3:3.0:2:1"
    "fmrgj_nfe41:jac:flow_map2:80:10:4:3:3.0:2:1"
    "fmrgj_nfe61:jac:flow_map2:112:14:7:3:3.0:2:1"
    "fmrgj_nfe81:jac:flow_map2:104:26:11:3:5.0:2:1"
    "fmrgj_nfe100:jac:flow_map2:208:26:11:3:5.0:2:1"
    "fmrge_nfe5:euc:flow_map1:8:5:0:1:0.7:0:3"
    "fmrge_nfe9:euc:flow_map1:8:5:2:2:0.7:0:3"
    "fmrge_nfe18:euc:flow_map2:32:8:0:1:0.7:2:3"
    "fmrge_nfe30:euc:flow_map1:29:21:3:3:0.3:0:3"
    "fmrge_nfe59:euc:flow_map2:120:15:6:3:0.7:2:3"
    "fmrge_nfe99:euc:flow_map2:200:25:11:3:0.7:2:3"
)

for cfg in "${CONFIGS[@]}"; do
    IFS=: read -r tag grad_mode sample_mode num_steps early_stop warmup_steps warmup_particles step_size unguided_steps num_optim_iters <<< "$cfg"
    NORMALIZE_FLAG=""
    [[ "$grad_mode" == "jac" ]] && NORMALIZE_FLAG="--normalize_grad"

    for SEED in $SEEDS; do
        OUTPUT_DIR="$OUTPUT_ROOT/${tag}_seed${SEED}"
        if [[ -f "$OUTPUT_DIR/results.jsonl" ]]; then
            echo "SKIP $tag seed=$SEED"
            continue
        fi
        echo ">> $tag seed=$SEED"
        python scripts/generate_geneval.py \
            --grad_mode "$grad_mode" \
            $NORMALIZE_FLAG \
            --sample_mode "$sample_mode" \
            --num_steps "$num_steps" \
            --early_stop "$early_stop" \
            --warmup_steps "$warmup_steps" \
            --warmup_particles "$warmup_particles" \
            --step_size "$step_size" \
            --unguided_steps "$unguided_steps" \
            --num_optim_iters "$num_optim_iters" \
            --grad_checkpointing \
            --resolution 512 \
            --seed "$SEED" \
            --output_dir "$OUTPUT_DIR" \
            --prompts_file "$PROMPTS_FILE" \
            --start_idx "$START_IDX" \
            --end_idx "$END_IDX" \
            --num_samples 1
    done
done
