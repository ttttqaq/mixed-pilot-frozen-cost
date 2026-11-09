#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SWE_NEO_PREPARED_IMAGES="/Users/liangjianfeng/Data/Router/SWE-neo-data/environment-images.json"
# Credentials stay in the invoking terminal environment, never in this script.
if [[ " ${*:-} " != *" --dry-run "* ]]; then
  : "${DEEPSEEK_API_KEY:?Please export DEEPSEEK_API_KEY in this terminal first}"
fi
exec /Users/liangjianfeng/miniconda3/envs/opensquilla/bin/python -u scripts/infer.py \
  --dataset run_0929/swe-neo-deepseek-flash-0929/provenance/replacement-inference-22/dataset \
  --split train --mode single-model --models deepseek-flash \
  --stop-mode run-mode --runs 1 --step-limit 50 --seed 42 \
  --config harness.json \
  --pricing run_0929/swe-neo-deepseek-flash-0929/provenance/replacement-inference-22/inference.prices.json \
  --cpus 4 --memory 8g --eval-timeout 3600 \
  --output-dir run_1002/swe-neo-deepseek-flash-replacements22 \
  "$@"
