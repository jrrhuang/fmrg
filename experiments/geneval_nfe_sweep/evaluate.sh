#!/bin/bash
set -e

if [[ -z "${GENEVAL_REPO:-}" || -z "${GENEVAL_MODELS:-}" ]]; then
    echo "ERROR: set GENEVAL_REPO and GENEVAL_MODELS"
    exit 1
fi

RESULTS_ROOT="${1:?usage: evaluate.sh <results_root>}"
[[ ! -d "$RESULTS_ROOT" ]] && { echo "ERROR: $RESULTS_ROOT not found"; exit 1; }

find "$RESULTS_ROOT" -mindepth 1 -maxdepth 4 -type d -name 'samples' -prune | while read samples_dir; do
    cfg_dir="$( dirname "$( dirname "$samples_dir" )" )"
    out="$cfg_dir/results.jsonl"
    if [[ -s "$out" ]]; then continue; fi
    echo ">> $cfg_dir"
    python "$GENEVAL_REPO/evaluation/evaluate_images.py" \
        "$cfg_dir" --outfile "$out" --model-path "$GENEVAL_MODELS"
    python "$GENEVAL_REPO/evaluation/summary_scores.py" "$out" \
        | tee "$cfg_dir/summary.txt"
done
