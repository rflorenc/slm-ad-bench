#!/usr/bin/env python

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

import torch

project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from core.inference import InferenceManager
from core.inference.local_transformers import LocalTransformersConfig
from core.inference.factory import InferenceBackendFactory
from core.evaluation import EvaluationEngine
from utils.file_utils import init_run_dir, write_run_info, combine_results
from utils.json_utils import safe_json_dump, ensure_json_serializable

class ModelConfig:
    def __init__(self, name, model_name, supported_backends, preferred_backend, backend_configs):
        self.name = name
        self.model_name = model_name
        self.supported_backends = supported_backends
        self.preferred_backend = preferred_backend
        self.backend_configs = backend_configs
    
    def get_backend_config(self, backend_name):
        return self.backend_configs.get(backend_name, {})

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def setup_hf_token():
    """Load and set Hugging Face token from file"""
    # Look for token file relative to script location
    script_dir = Path(__file__).parent
    token_file = script_dir / "hf_token.txt"
    if token_file.exists():
        try:
            with open(token_file) as f:
                token = f.read().strip()
            if token:
                os.environ["HUGGINGFACE_HUB_TOKEN"] = token
                logger.info("Hugging Face token loaded successfully")
            else:
                logger.warning("HF token file is empty")
        except Exception as e:
            logger.error(f"Error loading HF token: {e}")
    else:
        logger.warning(f"HF token file not found at: {token_file}")

def load_model_configs() -> Dict[str, ModelConfig]:
    """Load model configurations from YAML"""
    import yaml
    
    config_path = Path(__file__).parent / "config" / "models.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    configs = {}
    for model_data in config_data['models']:
        configs[model_data['name']] = ModelConfig(
            name=model_data['name'],
            model_name=model_data['model_name'],
            supported_backends=model_data['supported_backends'],
            preferred_backend=model_data['preferred_backend'],
            backend_configs=model_data['backend_configs']
        )
    
    return configs

def load_approaches(test_config: str = None) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
    """Load approaches and reasoning config from YAML configuration"""
    import yaml
    
    config_path = Path(__file__).parent / "config" / "approaches.yaml"
    with open(config_path) as f:
        config_data = yaml.safe_load(f)
    
    reasoning_config = config_data.get('reasoning_enhancements', {})
    
    if test_config and test_config in config_data.get('test_configs', {}):
        enabled_names = config_data['test_configs'][test_config]
        approaches = [
            {"name": approach['name'], "type": approach['type'], "model_name": approach['model_name']}
            for approach in config_data['approaches']
            if approach['name'] in enabled_names
        ]
        logger.info(f"Using test config '{test_config}' with {len(approaches)} approaches")
        return approaches, reasoning_config
    
    approaches = [
        {"name": approach['name'], "type": approach['type'], "model_name": approach['model_name']}
        for approach in config_data['approaches']
        if approach.get('enabled', True)
    ]
    
    logger.info(f"Loaded {len(approaches)} enabled approaches from configuration")
    return approaches, reasoning_config

APPROACHES = []
REASONING_CONFIG = {}

async def run_single_approach(approach_idx: int, data_args: List[str], nlines: int, dataset_type: str, 
                            contamination: str = "auto", mode: str = "labeled", global_backend: str = None) -> bool:
    """
    Run a single approach using backend
    
    This replaces the subprocess calls in the original run_all_models.py
    """
    if approach_idx >= len(APPROACHES):
        logger.error(f"Invalid approach index: {approach_idx}")
        return False
    
    approach = APPROACHES[approach_idx]
    safe_name = approach["name"].replace(" ", "_").replace("/", "_")
    
    logger.info(f"[{approach['name']}] Starting evaluation...")
    
    try:
        model_configs = load_model_configs()
        if approach["name"] not in model_configs:
            logger.error(f"No configuration found for {approach['name']}")
            return False
        
        model_config = model_configs[approach["name"]]
        
        # Determine backend to use (global override > approach config > default)
        if global_backend:
            backend_name = global_backend
            logger.info(f"[{approach['name']}] Using global backend override: {backend_name}")
        else:
            backend_name = approach.get("backend", "local_transformers")
            logger.info(f"[{approach['name']}] Using configured backend: {backend_name}")
        
        # Get backend-specific configuration
        backend_config = model_config.get_backend_config(backend_name)
        
        # Create appropriate config object
        if backend_name == "local_transformers":
            inference_config = LocalTransformersConfig(
                model_name=approach["model_name"],
                **backend_config
            )
        else:
            # Use factory to create config for other backends
            try:
                backend_info = InferenceBackendFactory.get_backend_info(backend_name)
                config_class = backend_info["config_class"]
                inference_config = config_class(
                    model_name=approach["model_name"],
                    **backend_config
                )
            except ValueError as e:
                logger.error(f"Backend '{backend_name}' not available: {e}")
                logger.info(f"Falling back to local_transformers for {approach['name']}")
                backend_name = "local_transformers"
                backend_config = model_config.get_backend_config("local_transformers")
                inference_config = LocalTransformersConfig(
                    model_name=approach["model_name"],
                    **backend_config
                )
        
        manager = InferenceManager()
        evaluation_engine = EvaluationEngine(REASONING_CONFIG)
        
        async with manager.backend_context(backend_name, inference_config) as backend:
            if mode == "unlabeled":
                results = await evaluation_engine.run_unlabeled_evaluation(
                    backend, data_args[0], nlines, dataset_type, contamination, approach['name'], RESULTS_DIR
                )
            elif dataset_type == "unsw-nb15" and len(data_args) == 2:
                results = await evaluation_engine.run_unsw_nb15_evaluation(
                    backend, data_args[0], data_args[1], nlines, approach['name'], RESULTS_DIR
                )
            else:
                results = await evaluation_engine.run_standard_evaluation(
                    backend, data_args[0], nlines, dataset_type, approach['name'], RESULTS_DIR
                )
            
            results_file = Path(RESULTS_DIR) / f"results_{safe_name}.json"
            with open(results_file, 'w') as f:
                sanitized_results = ensure_json_serializable(results)
                safe_json_dump(sanitized_results, str(results_file), indent=2)
            
            logger.info(f"[{approach['name']}] Completed successfully with {len(results)} result(s)")
            return True
            
    except Exception as e:
        logger.error(f"[{approach['name']}] Failed: {e}")
        return False

async def main():
    setup_hf_token()

    if len(sys.argv) < 2 or (len(sys.argv) == 2 and sys.argv[1] in ["--help", "-h"]):
        logger.info("Usage:")
        logger.info("  labeled, UNSW (train + test): python run_benchmark.py [--backend BACKEND] train.csv test.csv n unsw-nb15")
        logger.info("  labeled, single file: python run_benchmark.py [--backend BACKEND] data.csv n [dataset_type]")
        logger.info("  unlabeled: python run_benchmark.py [--backend BACKEND] data n dataset_type unlabeled [contamination]")
        logger.info("  test configurations: python run_benchmark.py --test-config quick_test [--backend BACKEND] data n dataset_type unlabeled [contamination]")
        logger.info("  Available backends: local_transformers, vllm")
        return

    mode = "labeled"
    contamination = "auto"
    dataset_type = None
    data_args = []
    test_config = None
    
    # Global backend override applied to all models
    global_backend = None
    
    args = sys.argv[1:]
    if len(args) > 0 and args[0] == "--test-config":
        if len(args) < 2:
            logger.error("--test-config requires a configuration name")
            return
        test_config = args[1]
        args = args[2:]  # Remove --test-config and config name
        logger.info(f"Using test configuration: {test_config}")
    elif len(args) > 0 and args[0] == "--test-configs":
        if len(args) < 2:
            logger.error("--test-configs requires a configuration name")
            return
        test_config = args[1]
        args = args[2:]  # Remove --test-configs and config name
        logger.info(f"Using test configuration: {test_config}")

    if len(args) > 0 and args[0] == "--backend":
        if len(args) < 2:
            logger.error("--backend requires a backend name")
            return
        global_backend = args[1]
        args = args[2:]  # Remove --backend and backend name
        logger.info(f"Global backend override: {global_backend}")

        available_backends = InferenceBackendFactory.list_backends()
        if global_backend not in available_backends:
            logger.error(f"Backend '{global_backend}' not available. Available: {available_backends}")
            return

    if len(args) < 2:
        logger.error("Insufficient arguments after parsing flags")
        logger.info("Use --help for usage information")
        return

    if len(args) >= 4 and args[3] == "unlabeled":
        mode = "unlabeled"
        data_path = os.path.abspath(args[0])
        nlines = int(args[1])
        dataset_type = args[2]
        if len(args) > 4 and args[4] != "auto":
            contamination = args[4]
        else:
            contamination = "auto"

        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return
        data_args = [data_path]

    elif len(args) >= 4 and args[-1] == "unsw-nb15":
        train_path = os.path.abspath(args[0])
        test_path = os.path.abspath(args[1])
        nlines = int(args[2])

        if not (os.path.exists(train_path) and os.path.exists(test_path)):
            logger.error("Train/test file missing")
            return

        dataset_type = "unsw-nb15"
        data_args = [train_path, test_path]

    else:
        data_path = os.path.abspath(args[0])
        nlines = int(args[1])
        dataset_type = args[2] if len(args) > 2 else "eventtraces"

        if not os.path.exists(data_path):
            logger.error(f"Data file not found: {data_path}")
            return
        data_args = [data_path]

    run_label = f"{mode}_{dataset_type}"
    global RESULTS_DIR
    RESULTS_DIR = init_run_dir(run_type=run_label)

    if mode == "unlabeled":
        write_run_info(RESULTS_DIR, data_args[0], nlines, f"{dataset_type}_unlabeled")
    elif dataset_type == "unsw-nb15" and len(data_args) == 2:
        write_run_info(RESULTS_DIR, f"train:{data_args[0]}, test:{data_args[1]}", nlines, dataset_type)
    else:
        write_run_info(RESULTS_DIR, data_args[0], nlines, dataset_type)

    global APPROACHES
    global REASONING_CONFIG
    APPROACHES, REASONING_CONFIG = load_approaches(test_config)
    
    logger.info(f"APPROACHES: {len(APPROACHES)} models")
    logger.info(f"Results directory: {RESULTS_DIR}")
    
    successful_runs = 0
    
    for idx in range(len(APPROACHES)):
        logger.info(f"\n=== Running approach {idx+1}/{len(APPROACHES)}: {APPROACHES[idx]['name']} ===")
        
        success = await run_single_approach(
            idx, data_args, nlines, dataset_type, contamination, mode, global_backend
        )
        
        if success:
            successful_runs += 1
        else:
            logger.warning(f"Approach {idx} failed; continuing...")
        
        # long wait for GPU mem cleanup
        await asyncio.sleep(5)
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            if allocated > 1.0:  # More than 1GB still allocated
                logger.warning(f"High GPU memory usage before next model: {allocated:.2f} GB")
                logger.info("Forcing additional cleanup...")
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                await asyncio.sleep(3)
    
    logger.info("\nCombining results...")
    df = combine_results(RESULTS_DIR)
    if df is not None:
        logger.info(f"Final combined results shape: {df.shape}")
    
    logger.info(f"\nBenchmark completed!")
    logger.info(f"Successful runs: {successful_runs}/{len(APPROACHES)}")
    logger.info(f"Results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    asyncio.run(main())