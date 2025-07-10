#!/usr/bin/env bash
# Usage: bash benchmark.sh

set -euo pipefail
IFS=$'\n\t'

HF_TOKEN_FILE="hf_token.txt"
if [[ ! -f $HF_TOKEN_FILE ]]; then
  echo "Hugging Face token file not found: $HF_TOKEN_FILE"
  exit 1
fi
export HUGGINGFACE_HUB_TOKEN="$(<"$HF_TOKEN_FILE")"

LOGDIR="bench_logs/small"
mkdir -p "$LOGDIR"

timestamp() { date '+%d_%H%M'; }

TRACES_LBL="datasets/intermediate_tasks/task1_systems/loghub/HDFS/HDFS_v1_labelled/preprocessed/Event_traces.csv"

UNSW_TRAIN="datasets/intermediate_tasks/intrusion_detection/UNSW_NB15/UNSW_NB15_training-set.csv"
UNSW_TEST="datasets/intermediate_tasks/intrusion_detection/UNSW_NB15/UNSW_NB15_testing-set.csv"

TRACES_UL="datasets/intermediate_tasks/task1_systems/loghub/HDFS/HDFS_2k.log_structured.csv"
UNSW_UL="datasets/intermediate_tasks/intrusion_detection/UNSW_NB15/UNSW-NB15_1.csv"


run_and_log() {
  local tag="$1"; shift
  local log="$LOGDIR/${tag}__$(timestamp).log"
  echo "▶ $tag  →  $log"
  if "$@" &> "$log"; then
    echo "✔ $tag finished OK"
  else
    echo "✗ $tag failed  (see $log)"
  fi
}

echo "🚀 Running SLM-AD-BENCH small_models with backend architecture"
echo "Using local transformers inference with PyTorch"
echo ""

# Labeled benchmarks
ET_SIZES=(2000)              # edit e.g. (1000 5000 10000)
for N in "${ET_SIZES[@]}"; do
run_and_log "eventtraces_lbl_${N}k" \
  python run_benchmark.py --test-config small_models "$TRACES_LBL" ${N} eventtraces
done

UNSW_SIZES=(2000)              # edit e.g. (1000 5000 10000)
for N in "${UNSW_SIZES[@]}"; do
  run_and_log "unsw_lbl_${N}" \
    python run_benchmark.py --test-config small_models "$UNSW_TRAIN" "$UNSW_TEST" "$N" unsw-nb15
done

# Unlabeled benchmarks
ET_SIZES=(2000)              # edit e.g. (1000 5000 10000)
for N in "${ET_SIZES[@]}"; do
run_and_log "eventtraces_ul_${N}k_auto" \
  python run_benchmark.py --test-config small_models "$TRACES_UL" ${N} eventtraces unlabeled auto
done

UNSW_SIZES=(2000)              # edit e.g. (1000 5000 10000)
for N in "${UNSW_SIZES[@]}"; do
run_and_log "unsw_ul_${N}_auto" \
  python run_benchmark.py --test-config small_models "$UNSW_UL" ${N} unsw-nb15 unlabeled auto
done

echo ""
echo "Benchmark small_models completed!"
echo "Results saved in output_results/"
echo "📋 Logs available in $LOGDIR/"
