"""
Unit tests for inference backends
"""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch

from core.inference.base import InferenceConfig, EmbeddingResult, GenerationResult
from core.inference.local_transformers import LocalTransformersBackend, LocalTransformersConfig
from core.inference.factory import InferenceBackendFactory
from core.inference.manager import InferenceManager

class TestLocalTransformersConfig:
    """Test configuration class for local transformers"""
    
    def test_config_creation(self):
        """Test basic config creation"""
        config = LocalTransformersConfig(
            model_name="test-model",
            batch_size=2,
            max_length=128,
            use_4bit=True
        )
        
        assert config.model_name == "test-model"
        assert config.batch_size == 2
        assert config.max_length == 128
        assert config.use_4bit is True
        assert config.device == "cuda"  # default
    
    def test_rtx3060ti_optimized_config(self):
        """Test configuration optimized (8GB)"""
        # Small model config (1.5B-3B)
        small_config = LocalTransformersConfig(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            batch_size=4,
            use_4bit=True,
            use_nested_quant=True,
            use_cpu_offload=False
        )
        
        assert small_config.batch_size == 4  # Can use larger batches
        assert small_config.use_4bit is True  # Always use quantization
        assert small_config.use_cpu_offload is False  # No need for small models
        
        # Large model config (7B-8B) 
        large_config = LocalTransformersConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            batch_size=1,
            use_4bit=True,
            use_nested_quant=True,
            use_cpu_offload=True
        )
        
        assert large_config.batch_size == 1  # Small batches for large models
        assert large_config.use_cpu_offload is True  # Offload for 7B+ models

class TestLocalTransformersBackend:
    """Test local transformers backend implementation"""
    
    def test_backend_creation(self):
        """Test backend instantiation"""
        config = LocalTransformersConfig(model_name="test-model")
        backend = LocalTransformersBackend(config)
        
        assert backend.config.model_name == "test-model"
        assert backend.backend_name == "local_transformers"
        assert backend.supports_batch_inference is True
        assert backend.max_batch_size == 32
        assert backend.is_loaded is False
    
    @pytest.mark.asyncio
    async def test_context_managers(self):
        """Test async and sync context managers"""
        config = LocalTransformersConfig(model_name="test-model")
        
        with patch.object(LocalTransformersBackend, 'load_model') as mock_load:
            with patch.object(LocalTransformersBackend, 'unload_model') as mock_unload:
                mock_load.return_value = {"load_time": 1.0}
                
                # Test async context manager
                async with LocalTransformersBackend(config) as backend:
                    assert backend is not None
                    mock_load.assert_called_once()
                
                mock_unload.assert_called_once()
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForCausalLM.from_pretrained')
    @patch('torch.cuda.is_available')
    def test_model_loading_mocked(self, mock_cuda, mock_model, mock_tokenizer):
        """Test model loading with mocked transformers"""
        mock_cuda.return_value = True
        
        # Mock tokenizer
        mock_tok = Mock()
        mock_tok.pad_token = None
        mock_tok.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tok
        
        # Mock model
        mock_mod = Mock()
        mock_mod.config.hidden_size = 768
        mock_model.return_value = mock_mod
        
        config = LocalTransformersConfig(
            model_name="test-model",
            use_4bit=True,
            use_nested_quant=True
        )
        
        backend = LocalTransformersBackend(config)
        
        # Test model loading
        import asyncio
        stats = asyncio.run(backend.load_model())
        
        assert backend.is_loaded
        assert backend.tokenizer is not None
        assert backend.model is not None
        assert "load_time" in stats
        assert stats["quantization_enabled"] is True
    
    @pytest.mark.asyncio
    async def test_embedding_generation_mocked(self):
        """Test embedding generation with mocked model"""
        config = LocalTransformersConfig(model_name="test-model")
        backend = LocalTransformersBackend(config)
        
        # Mock the model and tokenizer
        backend.tokenizer = Mock()
        backend.model = Mock()
        backend.model.eval = Mock()
        backend.model.config.hidden_size = 768
        backend.is_loaded = True
        backend.device = torch.device("cpu")  # Use CPU for testing
        
        # Mock tokenizer output
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        backend.tokenizer.return_value = mock_inputs
        
        # Mock model output
        mock_outputs = Mock()
        mock_hidden = torch.randn(2, 10, 768)  # batch_size=2, seq_len=10, hidden_size=768
        mock_outputs.hidden_states = [mock_hidden]  # Last layer
        backend.model.return_value = mock_outputs
        
        texts = ["log entry 1", "log entry 2"]
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.no_grad'):
                result = await backend.generate_embeddings(texts)
        
        assert isinstance(result, EmbeddingResult)
        assert result.embeddings.shape == (2, 768)
        assert result.processing_time > 0
        assert result.metadata["backend"] == "local_transformers"
        assert result.metadata["samples_processed"] == 2
    
    @pytest.mark.asyncio
    async def test_memory_cleanup(self):
        """Test proper memory cleanup"""
        config = LocalTransformersConfig(model_name="test-model")
        backend = LocalTransformersBackend(config)
        
        # Mock model and tokenizer
        backend.model = Mock()
        backend.tokenizer = Mock()
        backend.is_loaded = True
        
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.empty_cache') as mock_empty:
                with patch('torch.cuda.synchronize') as mock_sync:
                    await backend.unload_model()
        
        assert backend.model is None
        assert backend.tokenizer is None
        assert backend.is_loaded is False
        mock_empty.assert_called_once()
        mock_sync.assert_called_once()

class TestInferenceBackendFactory:
    """Test the backend factory"""
    
    def test_create_local_transformers_backend(self):
        """Test creating local transformers backend"""
        config = LocalTransformersConfig(model_name="test-model")
        backend = InferenceBackendFactory.create_backend("local_transformers", config)
        
        assert isinstance(backend, LocalTransformersBackend)
        assert backend.backend_name == "local_transformers"
    
    def test_create_backend_with_dict_config(self):
        """Test creating backend with dictionary config"""
        config_dict = {
            "model_name": "test-model",
            "batch_size": 2,
            "use_4bit": True
        }
        
        backend = InferenceBackendFactory.create_backend("local_transformers", config_dict)
        
        assert isinstance(backend, LocalTransformersBackend)
        assert backend.config.model_name == "test-model"
        assert backend.config.batch_size == 2
        assert backend.config.use_4bit is True
    
    def test_unknown_backend_error(self):
        """Test error handling for unknown backend"""
        config = LocalTransformersConfig(model_name="test-model")
        
        with pytest.raises(ValueError, match="Unknown backend type"):
            InferenceBackendFactory.create_backend("unknown_backend", config)
    
    def test_list_available_backends(self):
        """Test listing available backends"""
        backends = InferenceBackendFactory.list_backends()
        assert "local_transformers" in backends

class TestInferenceManager:
    """Test the inference manager"""
    
    @pytest.mark.asyncio
    async def test_backend_context_manager(self):
        """Test backend context management"""
        config = LocalTransformersConfig(model_name="test-model")
        manager = InferenceManager()
        
        with patch.object(LocalTransformersBackend, 'load_model') as mock_load:
            with patch.object(LocalTransformersBackend, 'unload_model') as mock_unload:
                mock_load.return_value = {"load_time": 1.0}
                
                async with manager.backend_context("local_transformers", config) as backend:
                    assert manager.current_backend is backend
                    assert isinstance(backend, LocalTransformersBackend)
                
                # After context exit
                assert manager.current_backend is None
                mock_unload.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_fallback_mechanism(self):
        """Test fallback to secondary backend"""
        primary_config = LocalTransformersConfig(model_name="failing-model")
        fallback_config = LocalTransformersConfig(model_name="working-model")
        
        manager = InferenceManager()
        
        with patch.object(InferenceBackendFactory, 'create_backend') as mock_create:
            # First backend fails to load
            failing_backend = Mock()
            failing_backend.load_model.side_effect = Exception("Model not found")
            
            # Second backend works
            working_backend = Mock()
            working_backend.load_model.return_value = {"load_time": 1.0}
            working_backend.generate_embeddings.return_value = EmbeddingResult(
                embeddings=np.random.randn(1, 768),
                metadata={"backend": "local_transformers", "fallback_used": True, "backend_attempt": 2},
                processing_time=1.0
            )
            working_backend.unload_model.return_value = None
            
            mock_create.side_effect = [failing_backend, working_backend]
            
            result = await manager.generate_embeddings_with_fallback(
                texts=["test text"],
                primary_backend="local_transformers",
                primary_config=primary_config,
                fallback_backends=[("local_transformers", fallback_config)]
            )
            
            assert result.metadata["fallback_used"] is True
            assert result.metadata["backend_attempt"] == 2
    
    def test_get_backend_stats(self):
        """Test getting backend statistics"""
        manager = InferenceManager()
        manager.backend_stats["local_transformers"] = {"load_time": 1.5}
        
        stats = manager.get_backend_stats()
        assert stats["local_transformers"]["load_time"] == 1.5
        
        # Should return copy, not reference
        stats["local_transformers"]["load_time"] = 2.0
        assert manager.backend_stats["local_transformers"]["load_time"] == 1.5

class TestRTX3060TiOptimizations:
    """Test specific optimizations  (8GB VRAM)"""
    
    def test_memory_conservative_configs(self):
        """Test that configs are conservative for 8GB VRAM"""
        # Test small model config
        small_config = LocalTransformersConfig(
            model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            batch_size=4,
            use_4bit=True
        )
        
        backend = LocalTransformersBackend(small_config)
        # Should allow reasonable batch size for small models
        assert backend.config.batch_size <= 4
        assert backend.max_batch_size <= 32
        
        # Test large model config  
        large_config = LocalTransformersConfig(
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
            batch_size=1,
            use_4bit=True,
            use_cpu_offload=True
        )
        
        large_backend = LocalTransformersBackend(large_config)
        # Should use minimal batch size and CPU offload for large models
        assert large_backend.config.batch_size == 1
        assert large_backend.config.use_cpu_offload is True
    
    @patch('torch.cuda.is_available')
    @patch('torch.cuda.get_device_properties')
    def test_gpu_memory_detection(self, mock_props, mock_cuda):
        """Test GPU memory detection"""
        mock_cuda.return_value = True
        
        # Mock RTX 3060ti properties
        mock_device = Mock()
        mock_device.total_memory = 8 * 1024**3  # 8GB
        mock_device.name = "NVIDIA GeForce RTX 3060 Ti"
        mock_props.return_value = mock_device
        
        config = LocalTransformersConfig(
            model_name="test-model",
            device="cuda"
        )
        
        backend = LocalTransformersBackend(config)
        
        if torch.cuda.is_available():
            assert str(backend.device) == "cuda"
        else:
            assert str(backend.device) == "cpu"
    
    def test_quantization_enabled_by_default(self):
        """Test that quantization is enabled for memory efficiency"""
        config = LocalTransformersConfig(
            model_name="test-model",
            use_4bit=True,
            use_nested_quant=True
        )
        
        backend = LocalTransformersBackend(config)
        assert config.use_4bit is True
        assert config.use_nested_quant is True
        assert config.torch_dtype == "float16"  # Memory efficient dtype

class TestErrorHandling:
    """Test error handling and robustness"""
    
    @pytest.mark.asyncio
    async def test_model_loading_failure(self):
        """Test handling of model loading failures"""
        config = LocalTransformersConfig(model_name="non-existent-model")
        backend = LocalTransformersBackend(config)
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
            mock_tokenizer.side_effect = Exception("Model not found")
            
            with pytest.raises(Exception):
                await backend.load_model()
            
            assert backend.is_loaded is False
    
    @pytest.mark.asyncio  
    async def test_embedding_generation_failure_fallback(self):
        """Test fallback behavior when embedding generation fails"""
        config = LocalTransformersConfig(model_name="test-model")
        backend = LocalTransformersBackend(config)
        
        # Setup mock model that fails on batch processing but works on single items
        backend.tokenizer = Mock()
        backend.model = Mock()
        backend.model.eval = Mock()
        backend.model.config.hidden_size = 768
        backend.is_loaded = True
        backend.device = torch.device("cpu")
        
        # Mock tokenizer
        mock_inputs = Mock()
        mock_inputs.to.return_value = mock_inputs
        backend.tokenizer.return_value = mock_inputs
        
        # Mock model to fail on first call, succeed on individual calls
        call_count = 0
        def mock_model_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call (batch) fails
                raise Exception("CUDA out of memory")
            else:
                # Individual calls succeed
                mock_outputs = Mock()
                mock_hidden = torch.randn(1, 10, 768)
                mock_outputs.hidden_states = [mock_hidden]
                return mock_outputs
        
        backend.model.side_effect = mock_model_call
        
        texts = ["text1", "text2"]
        
        with patch('torch.cuda.is_available', return_value=False):
            with patch('torch.no_grad'):
                # Should fall back to individual processing
                result = await backend.generate_embeddings(texts)
        
        assert result.embeddings.shape[0] == 2  # Both texts processed
        assert call_count > 1  # Multiple calls due to fallback

if __name__ == "__main__":
    pytest.main([__file__])