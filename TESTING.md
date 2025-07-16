# Testing Guide for SLM-AD-BENCH

## Quick Validation Testing

After making changes to the codebase, you can quickly validate that everything still works using the provided test scripts.

### benchmarks/benchmark-test-quick.sh

A fast testing script that runs all 4 evaluation approaches with configurable parameters:

```bash
# Basic usage (1000 lines, 2 models)
cd benchmarks && ./benchmark-test-quick.sh

# Custom number of lines (500 lines, 2 models)
cd benchmarks && ./benchmark-test-quick.sh 500

# Single model quick test
cd benchmarks && ./benchmark-test-quick.sh 1000 true

# Minimal test (200 lines, single model)
cd benchmarks && ./benchmark-test-quick.sh 200 true
```

### Test Configurations

The script uses test configurations from `config/approaches.yaml`:

- **test**: 2 models (Granite_3.2-2b-instruct, DeepSeek-R1-Distill-Qwen-1.5B)
- **test_single**: 1 model (DeepSeek-R1-Distill-Qwen-1.5B) - fastest option

### Test Coverage

The quick test runs all 4 evaluation approaches:

1. **Labeled EventTraces** - Standard supervised evaluation
2. **Labeled UNSW-NB15** - Network intrusion detection with train/test split
3. **Unlabeled EventTraces** - Unsupervised anomaly detection
4. **Unlabeled UNSW-NB15** - Unsupervised network intrusion detection

### Expected Runtime

- **Single model (200 lines)**: ~5-10 minutes
- **Single model (1000 lines)**: ~15-30 minutes
- **Dual model (1000 lines)**: ~30-60 minutes

### Validation Checklist

After running the test, check:

- [ ] All 4 test approaches completed successfully
- [ ] Results saved to `output_results/`
- [ ] Logs available in `bench_logs/test/`
- [ ] No Python errors or crashes
- [ ] Memory usage reasonable (monitor with `htop`)

### Troubleshooting

If tests fail:

1. Check logs in `bench_logs/test/`
2. Verify HuggingFace token in `hf_token.txt`
3. Ensure datasets are present in `datasets/` directory
4. Check GPU memory with `nvidia-smi`
5. Verify Python environment has all dependencies

### Original Test Scripts

Benchmark scripts are now located in the `benchmarks/` directory:

- `benchmarks/benchmark-test.sh`: Original test script (2000 lines, 2 models)
- `benchmarks/benchmark-small.sh`: Small benchmark (2000 lines, small models)
- `benchmarks/benchmark-large.sh`: Large benchmark (2000 lines, large models)
- `benchmarks/benchmark.sh`: Full benchmark (2000 lines, all models)
- `benchmarks/benchmark-test-vllm.sh`: vLLM backend test (2000 lines, 2 models)

Usage:
```bash
cd benchmarks && ./benchmark-test.sh
cd benchmarks && ./benchmark-small.sh
cd benchmarks && ./benchmark-large.sh
```