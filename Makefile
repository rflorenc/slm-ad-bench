.PHONY: help test test-vllm venv-test venv-clean install-dev lint format clean

PYTHON := python3
VENV := venv_test
VENV_PYTHON := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

help: 
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

install-dev:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/ -m "not gpu and not slow" --tb=short

test-all:
	pytest tests/ --tb=short

test-vllm: ## requires vLLM
	pytest tests/test_vllm_performance_comparison.py -v -s


$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip setuptools wheel

venv-install: $(VENV)
	$(VENV_PIP) install -r requirements.txt

venv-test: venv-install
	@echo "Testing in isolated virtual environment..."
	$(VENV_PYTHON) -m pytest tests/ -m "not gpu and not slow" --tb=short
	@echo "Virtual environment test completed successfully!"

venv-test-vllm: venv-install
	@echo "Testing vLLM performance comparison in isolated virtual environment..."
	@echo "Running as standalone script to see detailed output..."
	$(VENV_PYTHON) tests/test_vllm_performance_comparison.py
	@echo "vLLM test completed!"

venv-test-vllm-pytest: venv-install
	@echo "Testing vLLM performance comparison via pytest..."
	$(VENV_PYTHON) -m pytest tests/test_vllm_performance_comparison.py -v -s -m "gpu and slow"
	@echo "vLLM pytest test completed!"

venv-check-conflicts: $(VENV)
	@echo "Checking for dependency conflicts..."
	$(VENV_PIP) install -r requirements.txt
	@echo "Main requirements installed successfully"
	$(VENV_PIP) check

venv-benchmark: venv-install
	@echo "Running benchmark test in virtual environment..."
	cd $(VENV) && ../$(VENV_PYTHON) ../run_benchmark.py --test-config test ../datasets/intermediate_tasks/task1_systems/loghub/HDFS/HDFS_v1_labelled/preprocessed/EventSequences.csv 100 eventtraces


lint:
	black --check --diff .
	isort --check-only --diff .
	flake8 .

format:
	black .
	isort .


venv-clean:
	rm -rf $(VENV)

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/ dist/ .pytest_cache/ .coverage htmlcov/
	rm -f vllm_performance_comparison_results.json

deps-status:
	@echo "=== Current Python Environment ==="
	python --version
	pip list | grep -E "(torch|transformers|vllm|bitsandbytes|accelerate)"