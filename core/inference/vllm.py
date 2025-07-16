"""
vLLM inference backend implementation
"""

import os
import time
import logging
import asyncio
import numpy as np
import warnings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Suppress vLLM progress bars and verbose output
os.environ["VLLM_LOGGING_LEVEL"] = "WARNING"
os.environ["VLLM_DISABLE_LOGGING"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"

# Suppress tqdm progress bars from vLLM
import tqdm
original_tqdm = tqdm.tqdm

class NoOpTqdm:
    def __init__(self, *args, **kwargs):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass
    def update(self, *args):
        pass
    def close(self):
        pass
    def set_description(self, *args):
        pass

# Replace tqdm only for vLLM operations
def disable_tqdm_for_vllm():
    tqdm.tqdm = NoOpTqdm
    tqdm.tqdm.tqdm = NoOpTqdm

def restore_tqdm():
    tqdm.tqdm = original_tqdm

from .base import InferenceBackend, InferenceConfig, EmbeddingResult, GenerationResult

logger = logging.getLogger(__name__)

@dataclass
class VLLMConfig(InferenceConfig):
    """Configuration for vLLM backend"""
    # Memory and performance settings
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 512
    enforce_eager: bool = True
    max_num_seqs: int = 1
    
    # Quantization settings
    use_quantization: bool = True
    quantization_method: str = "auto"  # "auto", "bitsandbytes", "awq", "gptq"
    
    # Generation settings
    temperature: float = 0.7
    top_p: float = 0.95

class VLLMBackend(InferenceBackend):
    """vLLM inference backend with automatic memory optimization"""
    
    backend_name = "vllm"
    supports_batch_inference = True
    max_batch_size = 32  # Will be adjusted based on model size
    
    def __init__(self, config: VLLMConfig):
        super().__init__(config)
        self.config: VLLMConfig = config
        self.llm = None
        self.embedding_llm = None
        # Using single model instance for both embedding and text generation
        self._is_large_model = None
        
        # Fix Intel MKL threading conflicts
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1" 
        os.environ["MKL_THREADING_LAYER"] = "INTEL"
        
    @property
    def model(self):
        """Return the embedding model for compatibility with RAG system"""
        return self.embedding_llm
    
    @model.setter
    def model(self, value):
        """Setter for compatibility with base class initialization"""
        # The base class sets self.model = None, but we use embedding_llm instead
        # This setter allows the base class to work without breaking our design
        if value is None:
            # Allow base class to set None during initialization
            pass
        else:
            # For any other values, we don't actually use them since we manage embedding_llm separately
            pass
    
    @property
    def is_large_model(self) -> bool:
        """Detect if this is a large model requiring aggressive optimization"""
        if self._is_large_model is None:
            model_name_lower = self.config.model_name.lower()
            self._is_large_model = any(size in model_name_lower for size in ['7b', '8b', '13b', '70b'])
        return self._is_large_model
    
    async def load_model(self) -> Dict[str, Any]:
        """Load vLLM model with automatic optimization"""
        start_time = time.time()
        
        try:
            from vllm import LLM, SamplingParams, PoolingParams
        except ImportError as e:
            raise ImportError("vLLM not available. Install with: pip install vllm>=0.6.0") from e
        
        logger.info(f"Loading vLLM model: {self.config.model_name}")
        logger.info(f"Large model detected: {self.is_large_model}")
        
        # Base parameters for all configurations
        base_params = {
            "model": self.config.model_name,
            "dtype": self.config.dtype,
            "enforce_eager": self.config.enforce_eager
        }
        
        # Add quantization for large models
        if self.config.use_quantization and self.is_large_model:
            quantization = self._get_best_quantization()
            if quantization:
                base_params["quantization"] = quantization
                logger.info(f"Using {quantization} quantization for large model")
        
        # Memory-optimized parameters based on model size
        if self.is_large_model:
            # Conservative settings for large models
            memory_params = {
                "max_model_len": min(self.config.max_model_len, 512),
                "gpu_memory_utilization": min(self.config.gpu_memory_utilization, 0.7),
                "max_num_seqs": 1,
                "max_num_batched_tokens": 512  # Match max_model_len * max_num_seqs
            }
        else:
            # Standard settings for small models
            max_model_len = self.config.max_model_len
            max_num_seqs = self.config.max_num_seqs
            memory_params = {
                "max_model_len": max_model_len,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_num_seqs": max_num_seqs,
                "max_num_batched_tokens": max_model_len * max_num_seqs  # Prevent warning
            }
        
        # Load embedding model (for 95% of workload)
        # Initialize with task="embed" for embedding generation
        try:
            disable_tqdm_for_vllm()
            embedding_params = {**base_params, **memory_params, "task": "embed"}
            self.embedding_llm = LLM(**embedding_params)
            restore_tqdm()
            logger.info("vLLM embedding model loaded successfully")
        except Exception as e:
            restore_tqdm()
            logger.error(f"Failed to load vLLM embedding model: {e}")
            raise
        
        self.is_loaded = True
        load_time = time.time() - start_time
        
        logger.info(f"vLLM model loaded in {load_time:.2f}s")
        return {"load_time": load_time, "backend": "vllm", "quantization": base_params.get("quantization")}
    
    def _get_best_quantization(self) -> Optional[str]:
        """Try quantization methods in order of preference"""
        methods = ["bitsandbytes", "awq", "gptq"]
        
        if self.config.quantization_method != "auto":
            methods = [self.config.quantization_method]
        
        for method in methods:
            try:
                # Quick validation test (without loading model)
                logger.debug(f"Testing {method} quantization compatibility...")
                return method
            except Exception as e:
                logger.warning(f"{method} quantization failed: {e}")
                continue
        
        logger.warning("No quantization method available, proceeding without quantization")
        return None
    
    async def generate_embeddings(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Generate embeddings using vLLM"""
        if not self.is_loaded or self.embedding_llm is None:
            raise RuntimeError("Model not loaded for embedding generation")
        
        start_time = time.time()
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        chunk_size = kwargs.get('chunk_size', 200)
        
        logger.info(f"Generating embeddings for {len(texts)} texts with vLLM (batch_size={batch_size})")
        
        try:
            from vllm import PoolingParams
            pooling_params = PoolingParams()
            all_embeddings = []
            
            # Process in chunks
            for chunk_start in range(0, len(texts), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(texts))
                chunk_texts = texts[chunk_start:chunk_end]
                
                # Process chunk in batches
                chunk_embeddings = []
                for i in range(0, len(chunk_texts), batch_size):
                    batch_texts = chunk_texts[i:i + batch_size]
                    
                    # Run in executor to avoid blocking async loop
                    loop = asyncio.get_event_loop()
                    outputs = await loop.run_in_executor(
                        None, 
                        lambda: self.embedding_llm.encode(batch_texts, pooling_params=pooling_params, use_tqdm=False)
                    )
                    
                    # Extract embeddings using vLLM 0.9.1+ API
                    batch_embeds = []
                    
                    # Get model's expected hidden size to match local_transformers
                    model_config = self.embedding_llm.llm_engine.model_config
                    expected_hidden_size = getattr(model_config.hf_config, 'hidden_size', 768)
                    
                    for output in outputs:
                        embedding_extracted = False
                        
                        # Try different ways to extract embeddings from vLLM output
                        if hasattr(output, 'outputs') and hasattr(output.outputs, 'data'):
                            try:
                                embedding_data = output.outputs.data
                                if hasattr(embedding_data, 'cpu'):
                                    embedding_data = embedding_data.cpu().numpy()
                                elif isinstance(embedding_data, np.ndarray):
                                    pass  # Already numpy
                                else:
                                    embedding_data = np.array(embedding_data)
                                
                                # Ensure correct shape
                                if embedding_data.ndim > 1:
                                    embedding_data = embedding_data.squeeze()
                                
                                # Check dimensions
                                current_dim = embedding_data.shape[0] if embedding_data.ndim >= 1 else 1
                                
                                if current_dim == expected_hidden_size:
                                    embedding_extracted = True
                                elif current_dim > 0:
                                    logger.warning(f"vLLM embedding dimension {current_dim} != expected {expected_hidden_size}, adjusting")
                                    if current_dim < expected_hidden_size:
                                        # Pad with zeros
                                        padding = np.zeros(expected_hidden_size - current_dim)
                                        embedding_data = np.concatenate([embedding_data, padding])
                                    else:
                                        # Truncate to expected size
                                        embedding_data = embedding_data[:expected_hidden_size]
                                    embedding_extracted = True
                            except Exception as e:
                                logger.debug(f"Failed to extract embedding from output.outputs.data: {e}")
                        
                        # Try alternative extraction methods
                        if not embedding_extracted and hasattr(output, 'embeddings'):
                            try:
                                embedding_data = output.embeddings
                                if hasattr(embedding_data, 'cpu'):
                                    embedding_data = embedding_data.cpu().numpy()
                                embedding_extracted = True
                            except Exception as e:
                                logger.debug(f"Failed to extract from output.embeddings: {e}")
                        
                        if not embedding_extracted:
                            # Critical: Log this failure, don't silently use random
                            logger.error(f"vLLM embedding extraction completely failed for text. This model may not support embeddings with vLLM.")
                            raise RuntimeError("vLLM embedding extraction failed. Consider using local_transformers backend instead.")
                        
                        batch_embeds.append(embedding_data)
                    
                    batch_embeds = np.array(batch_embeds)
                    chunk_embeddings.append(batch_embeds)
                
                if chunk_embeddings:
                    chunk_array = np.vstack(chunk_embeddings)
                    all_embeddings.append(chunk_array)
                
                logger.debug(f"Processed chunk {chunk_start}-{chunk_end} / {len(texts)}")
            
            embeddings = np.vstack(all_embeddings) if all_embeddings else np.zeros((len(texts), 768))
            processing_time = time.time() - start_time
            
            return EmbeddingResult(
                embeddings=embeddings,
                metadata={
                    "backend": "vllm",
                    "batch_size": batch_size,
                    "texts_processed": len(texts),
                    "embedding_dim": embeddings.shape[1],
                    "quantization": hasattr(self, '_quantization') and self._quantization,
                    "is_large_model": self.is_large_model
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"vLLM embedding generation failed: {e}")
            raise
    
    async def generate_text(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text using temporary model with memory management"""
        start_time = time.time()
        max_new_tokens = kwargs.get('max_new_tokens', 32)
        temperature = kwargs.get('temperature', self.config.temperature)
        
        try:
            from vllm import SamplingParams, LLM
            import gc
            import torch
            
            sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_new_tokens,
                top_p=self.config.top_p
            )
            
            # Temporarily unload embedding model to free memory
            logger.info("Temporarily unloading embedding model for text generation")
            if self.embedding_llm is not None:
                del self.embedding_llm
                self.embedding_llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            
            # Base parameters for generation model
            base_params = {
                "model": self.config.model_name,
                "dtype": self.config.dtype,
                "enforce_eager": self.config.enforce_eager
            }
            
            # Use conservative memory settings
            gen_params = {
                "max_model_len": 256,
                "gpu_memory_utilization": 0.5,  # Use available memory
                "max_num_seqs": 1,
                "max_num_batched_tokens": 256
            }
            
            disable_tqdm_for_vllm()
            generation_params = {**base_params, **gen_params, "task": "generate"}
            temp_generation_llm = LLM(**generation_params)
            logger.info("Temporary vLLM generation model loaded")
            
            loop = asyncio.get_event_loop()
            outputs = await loop.run_in_executor(
                None,
                lambda: temp_generation_llm.generate([prompt], sampling_params, use_tqdm=False)
            )
            restore_tqdm()
            
            generated_text = outputs[0].outputs[0].text
            
            # Clean up generation model
            del temp_generation_llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Reload embedding model
            logger.info("Reloading embedding model")
            disable_tqdm_for_vllm()
            embedding_params = {
                "model": self.config.model_name,
                "dtype": self.config.dtype,
                "enforce_eager": self.config.enforce_eager,
                "max_model_len": self.config.max_model_len,
                "gpu_memory_utilization": self.config.gpu_memory_utilization,
                "max_num_seqs": self.config.max_num_seqs,
                "max_num_batched_tokens": self.config.max_model_len * self.config.max_num_seqs,
                "task": "embed"
            }
            self.embedding_llm = LLM(**embedding_params)
            restore_tqdm()
            
            processing_time = time.time() - start_time
            
            return GenerationResult(
                text=generated_text,
                metadata={
                    "backend": "vllm",
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "is_large_model": self.is_large_model
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"vLLM text generation failed: {e}")
            raise
    
    
    async def unload_model(self) -> Dict[str, Any]:
        """Unload all models and free resources"""
        start_time = time.time()
        
        if self.embedding_llm is not None:
            del self.embedding_llm
            self.embedding_llm = None
        
        # Aggressive cleanup
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Multiple cleanup passes for fragmented memory
            for _ in range(3):
                torch.cuda.empty_cache()
                gc.collect()
        
        self.is_loaded = False
        unload_time = time.time() - start_time
        
        logger.info(f"vLLM model unloaded in {unload_time:.2f}s")
        return {"unload_time": unload_time}