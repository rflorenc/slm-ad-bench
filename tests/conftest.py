"""
provides shared test fixtures and configuration for testing
"""

import pytest
import asyncio
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import Mock

from core.inference.local_transformers import LocalTransformersConfig
from core.inference.base import EmbeddingResult, GenerationResult

@pytest.fixture
def rtx3060ti_small_model_config():
    """Configuration optimized for small models on RTX 3060ti"""
    return LocalTransformersConfig(
        model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        batch_size=4,
        max_length=128,
        use_4bit=True,
        use_nested_quant=True,
        use_cpu_offload=False,
        device="cuda",
        torch_dtype="float16"
    )

@pytest.fixture
def rtx3060ti_large_model_config():
    """Configuration optimized for large models on RTX 3060ti"""
    return LocalTransformersConfig(
        model_name="mistralai/Mistral-7B-Instruct-v0.3",
        batch_size=1,
        max_length=128,
        use_4bit=True,
        use_nested_quant=True,
        use_cpu_offload=True,
        device="cuda",
        torch_dtype="float16",
        low_cpu_mem_usage=True
    )

@pytest.fixture
def sample_log_texts():
    """Sample log texts for testing embedding generation"""
    return [
        "User login successful from IP 192.168.1.100",
        "Failed authentication attempt from IP 10.0.0.5", 
        "Database connection established successfully",
        "Error: Unable to connect to external service",
        "System startup completed in 2.3 seconds",
        "Warning: High memory usage detected (85%)",
        "Backup process initiated at 02:00 AM",
        "Network interface eth0 is down",
        "Configuration file reloaded successfully",
        "Application crashed with exit code 1"
    ]

@pytest.fixture
def mock_embedding_result():
    """Mock embedding result for testing"""
    def _create_result(num_samples=5, embedding_dim=768):
        return EmbeddingResult(
            embeddings=np.random.randn(num_samples, embedding_dim),
            metadata={
                "backend": "local_transformers",
                "samples_processed": num_samples,
                "batch_size": 2,
                "max_length": 128,
                "embedding_dim": embedding_dim
            },
            processing_time=1.5
        )
    return _create_result

@pytest.fixture
def mock_generation_result():
    """Mock text generation result for testing"""
    def _create_result(prompt="test prompt"):
        return GenerationResult(
            text="This appears to be a normal system operation with no security concerns detected.",
            metadata={
                "backend": "local_transformers",
                "prompt_length": len(prompt),
                "max_new_tokens": 100,
                "temperature": 0.7
            },
            processing_time=2.1
        )
    return _create_result

@pytest.fixture
def mock_backend():
    """Mock inference backend for testing"""
    backend = Mock()
    backend.backend_name = "local_transformers"
    backend.supports_batch_inference = True
    backend.max_batch_size = 32
    backend.is_loaded = False
    
    # Mock async methods
    async def mock_load():
        backend.is_loaded = True
        return {
            "load_time": 2.5,
            "model_size_params": 1500000,
            "device": "cuda", 
            "quantization_enabled": True
        }
    
    async def mock_unload():
        backend.is_loaded = False
    
    async def mock_embeddings(texts, **kwargs):
        return EmbeddingResult(
            embeddings=np.random.randn(len(texts), 768),
            metadata={
                "backend": "local_transformers",
                "samples_processed": len(texts),
                "batch_size": kwargs.get("batch_size", 2)
            },
            processing_time=1.5
        )
    
    async def mock_generate(prompt, **kwargs):
        return GenerationResult(
            text="Generated response text",
            metadata={
                "backend": "local_transformers",
                "prompt_length": len(prompt)
            },
            processing_time=2.0
        )
    
    backend.load_model = mock_load
    backend.unload_model = mock_unload
    backend.generate_embeddings = mock_embeddings
    backend.generate_text = mock_generate
    
    return backend

@pytest.fixture
def temp_results_dir():
    """Temporary directory for test results"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_unsw_nb15_data():
    """Sample UNSW-NB15 dataset for testing"""
    return {
        "train": [
            {"text": "normal network connection established", "label": 0},
            {"text": "legitimate user authentication", "label": 0},
            {"text": "standard data transfer operation", "label": 0},
            {"text": "suspicious connection from unknown IP", "label": 1},
            {"text": "potential intrusion attempt detected", "label": 1}
        ],
        "test": [
            {"text": "routine network maintenance", "label": 0},
            {"text": "malicious payload detected", "label": 1},
            {"text": "normal user session", "label": 0}
        ]
    }

@pytest.fixture
def sample_eventtraces_data():
    """Sample EventTraces dataset for testing"""
    return [
        {"text": "application startup sequence initiated", "label": 0},
        {"text": "normal operation cycle completed", "label": 0},
        {"text": "user input processed successfully", "label": 0},
        {"text": "error in critical system component", "label": 1},
        {"text": "unexpected application termination", "label": 1},
        {"text": "configuration file corrupted", "label": 1}
    ]

@pytest.fixture
def mock_results_data():
    """Mock evaluation results data"""
    return {
        "traditional_ml": [
            {
                "approach": "Granite_3.2-2b-instruct_traditional_binary",
                "classifier": "ExtraTrees_50_trees",
                "acc": 0.92,
                "f1": 0.89,
                "roc_auc": 0.94,
                "eval_type": "research_compliant_traditional",
                "methodology": "5_fold_cv_minmax_no_pca"
            }
        ],
        "llm_embeddings": [
            {
                "approach": "Granite_3.2-2b-instruct_llm", 
                "classifier": "ExtraTrees",
                "acc": 0.87,
                "f1": 0.84,
                "roc_auc": 0.91,
                "eval_type": "research_compliant_llm_embeddings",
                "methodology": "5_fold_cv_minmax_plus_pca",
                "embedding_time": 45.2,
                "backend": "local_transformers"
            }
        ]
    }

@pytest.fixture
def all_original_approaches():
    """List of all 9 original approaches for testing compatibility"""
    return [
        {"name": "Granite_3.3-2b-instruct", "model_name": "ibm-granite/granite-3.3-2b-instruct"},
        {"name": "Granite_3.2-2b-instruct", "model_name": "ibm-granite/granite-3.2-2b-instruct"},
        {"name": "Llama-3.2-3B-Instruct", "model_name": "meta-llama/Llama-3.2-3B-Instruct"},
        {"name": "DeepSeek-R1-Distill-Qwen-1.5B", "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"},
        {"name": "DeepSeek-R1-Distill-Qwen-7B", "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"},
        {"name": "Mistral-7B-Instruct-v0.3", "model_name": "mistralai/Mistral-7B-Instruct-v0.3"},
        {"name": "Granite_3.2_8B-instruct", "model_name": "ibm-granite/granite-3.2-8b-instruct"},
        {"name": "Llama-3.1-8B-Instruct", "model_name": "meta-llama/Llama-3.1-8B-Instruct"},
        {"name": "DeepSeek-R1-Distill-Llama-8B", "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"}
    ]

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Pytest hooks for custom behavior

def pytest_configure(config):
    """Configure pytest with custom settings"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "rtx3060ti: tests specific to RTX 3060ti hardware"
    )
    config.addinivalue_line(
        "markers", "memory_intensive: tests that use significant memory"
    )

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names"""
    for item in items:
        # Mark GPU-specific tests
        if "gpu" in item.nodeid.lower() or "rtx" in item.nodeid.lower():
            item.add_marker(pytest.mark.gpu)
        
        # Mark memory-intensive tests
        if "memory" in item.nodeid.lower() or "8gb" in item.nodeid.lower():
            item.add_marker(pytest.mark.memory_intensive)
        
        # Mark integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark e2e tests
        if "e2e" in item.nodeid:
            item.add_marker(pytest.mark.e2e)

@pytest.fixture
def skip_if_no_gpu():
    """Skip test if no GPU is available"""
    try:
        import torch
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
    except ImportError:
        pytest.skip("PyTorch not available")

@pytest.fixture
def gpu_memory_info():
    """Get GPU memory information for testing"""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            return {
                "total_memory_gb": props.total_memory / 1e9,
                "name": props.name,
                "compute_capability": f"{props.major}.{props.minor}"
            }
    except ImportError:
        pass
    
    return {
        "total_memory_gb": 8.0,  # Assume RTX 3060ti for testing
        "name": "NVIDIA GeForce RTX 3060 Ti",
        "compute_capability": "8.6"
    }