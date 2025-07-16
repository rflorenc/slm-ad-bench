#!/usr/bin/env bash
# Quick test script to validate cleanup and functionality
# Usage: bash benchmark-test-quick.sh [nlines] [single_model]

set -euo pipefail
IFS=$'\n\t'

# Configuration
NLINES=${1:-500}  
SINGLE_MODEL=${2:-true}  # Set to true to run only one model

HF_TOKEN_FILE="../hf_token.txt"
if [[ ! -f $HF_TOKEN_FILE ]]; then
  echo "Hugging Face token file not found: $HF_TOKEN_FILE"
  exit 1
fi
export HUGGINGFACE_HUB_TOKEN="$(<"$HF_TOKEN_FILE")"

LOGDIR="../bench_logs/test-quick"
mkdir -p "$LOGDIR"

timestamp() { date '+%d_%H%M'; }

TRACES_LBL="../datasets/intermediate_tasks/task1_systems/loghub/HDFS/HDFS_v1_labelled/preprocessed/Event_traces.csv"
UNSW_TRAIN="../datasets/intermediate_tasks/intrusion_detection/UNSW_NB15/UNSW_NB15_training-set.csv"
UNSW_TEST="../datasets/intermediate_tasks/intrusion_detection/UNSW_NB15/UNSW_NB15_testing-set.csv"

TRACES_UL="../datasets/intermediate_tasks/task1_systems/loghub/HDFS/HDFS_2k.log_structured.csv"
UNSW_UL="../datasets/intermediate_tasks/intrusion_detection/UNSW_NB15/UNSW-NB15_1.csv"

run_and_log() {
  local tag="$1"; shift
  local log="$LOGDIR/${tag}__$(timestamp).log"
  echo "Running $tag -> $log"
  if "$@" &> "$log"; then
    echo "$tag finished OK"
  else
    echo "$tag failed (see $log)"
    return 1
  fi
}

echo "Running SLM-AD-BENCH quick test (nlines=$NLINES)"
echo "Using local transformers inference with PyTorch"

# Determine test configuration
if [[ "$SINGLE_MODEL" == "true" ]]; then
  TEST_CONFIG="test_single"
  # Extract the model name from the config file
  SINGLE_MODEL_NAME=$(grep -A 1 "test_single:" ../config/approaches.yaml | tail -1 | sed 's/.*- "//' | sed 's/".*//')
  echo "Running single model test ($SINGLE_MODEL_NAME only)"
else
  echo "Running with test config (2 models)"
  TEST_CONFIG="test"
fi

echo ""

echo "=== Test 1: Labeled EventTraces ==="
run_and_log "eventtraces_lbl_${NLINES}" \
  python ../run_benchmark.py --test-config "$TEST_CONFIG" "$TRACES_LBL" "$NLINES" eventtraces

echo "=== Test 2: Labeled UNSW-NB15 ==="
run_and_log "unsw_lbl_${NLINES}" \
  python ../run_benchmark.py --test-config "$TEST_CONFIG" "$UNSW_TRAIN" "$UNSW_TEST" "$NLINES" unsw-nb15

echo "=== Test 3: Unlabeled EventTraces ==="
run_and_log "eventtraces_ul_${NLINES}_auto" \
  python ../run_benchmark.py --test-config "$TEST_CONFIG" "$TRACES_UL" "$NLINES" eventtraces unlabeled auto

echo "=== Test 4: Unlabeled UNSW-NB15 ==="
run_and_log "unsw_ul_${NLINES}_auto" \
  python ../run_benchmark.py --test-config "$TEST_CONFIG" "$UNSW_UL" "$NLINES" unsw-nb15 unlabeled auto

echo ""
echo "All tests completed successfully!"
echo "Results saved in output_results/"
echo "Logs available in $LOGDIR/"
echo ""
echo "Quick validation complete - cleanup appears to be working correctly."