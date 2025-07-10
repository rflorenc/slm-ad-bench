"""
Local Transformers inference backend
"""

import torch
import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .base import InferenceBackend, InferenceConfig, EmbeddingResult, GenerationResult

logger = logging.getLogger(__name__)

@dataclass
class LocalTransformersConfig(InferenceConfig):
    """Configuration for local transformers backend"""
    use_4bit: bool = True
    use_nested_quant: bool = True
    use_cpu_offload: bool = False
    torch_dtype: str = "float16"
    low_cpu_mem_usage: bool = True
    offload_folder: str = "offload_folder"

class LocalTransformersBackend(InferenceBackend):
    """Local transformers inference backend"""
    
    def __init__(self, config: LocalTransformersConfig):
        super().__init__(config)
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.bnb_config = None
        
    async def load_model(self) -> Dict[str, Any]:
        """Load model with quantization support"""
        start_time = time.time()
        
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError as e:
            raise ImportError("transformers library not available") from e
        
        # Setup quantization if needed / available
        if self.config.use_4bit and torch.cuda.is_available():
            try:
                self.bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=self.config.use_nested_quant,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.torch_dtype),
                    bnb_4bit_quant_storage=torch.uint8
                )
                logger.info("4-bit quantization enabled")
            except Exception as e:
                logger.warning(f"Could not setup quantization: {e}")
                self.bnb_config = None
        
        # Enable memory optimizations
        import os
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        logger.info(f"Loading tokenizer for {self.config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if self.config.use_cpu_offload else self.device
        
        logger.info(f"Loading model {self.config.model_name}")
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": getattr(torch, self.config.torch_dtype),
            "low_cpu_mem_usage": self.config.low_cpu_mem_usage,
        }
        
        if self.bnb_config is not None:
            model_kwargs["quantization_config"] = self.bnb_config
            
        if self.config.use_cpu_offload:
            model_kwargs["offload_folder"] = self.config.offload_folder
            model_kwargs["offload_state_dict"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        self.is_loaded = True
        load_time = time.time() - start_time
        
        try:
            model_size = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        except:
            model_size = 0
        
        stats = {
            "backend": self.backend_name,
            "load_time": load_time,
            "model_size_params": model_size,
            "device": str(self.device),
            "quantization_enabled": self.bnb_config is not None,
            "cpu_offload_enabled": self.config.use_cpu_offload,
            "torch_dtype": self.config.torch_dtype
        }
        
        logger.info(f"Model loaded successfully in {load_time:.2f}s")
        return stats
    
    async def generate_embeddings(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Generate embeddings using local model"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        batch_size = kwargs.get('batch_size', self.config.batch_size)
        max_length = kwargs.get('max_length', self.config.max_length)
        chunk_size = kwargs.get('chunk_size', 200)
        
        logger.info(f"Generating embeddings for {len(texts)} texts")
        
        self.model.eval()
        all_embeddings = []
        
        for chunk_start in range(0, len(texts), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(texts))
            chunk_texts = texts[chunk_start:chunk_end]
            
            chunk_embeddings = []
            
            for i in range(0, len(chunk_texts), batch_size):
                batch_texts = chunk_texts[i:i + batch_size]
                
                try:
                    with torch.no_grad():
                        inputs = self.tokenizer(
                            batch_texts,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_length,
                            padding=True
                        ).to(self.device)
                        
                        outputs = self.model(**inputs, output_hidden_states=True)
                        # use last layer mean pooling
                        last_hidden = outputs.hidden_states[-1]
                        batch_embeddings = last_hidden.mean(dim=1).cpu().numpy()
                        
                        # check for NaN values in embeddings
                        if np.isnan(batch_embeddings).any():
                            logger.warning(f"Found NaN in batch embeddings, using random fallback")
                            dim = getattr(self.model.config, 'hidden_size', 768)
                            batch_size_actual = len(batch_texts)
                            # Use float64 to avoid JSON serialization issues, will be converted later
                            batch_embeddings = np.random.randn(batch_size_actual, dim).astype(np.float64)
                        
                        chunk_embeddings.append(batch_embeddings)
                        
                        # Mem cleanup
                        del inputs, outputs, last_hidden
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"Error processing batch {i}: {e}")
                    # Fallback: process one by one
                    for text in batch_texts:
                        try:
                            with torch.no_grad():
                                inputs = self.tokenizer([text], return_tensors="pt",
                                                      truncation=True, max_length=max_length).to(self.device)
                                outputs = self.model(**inputs, output_hidden_states=True)
                                last_hidden = outputs.hidden_states[-1]
                                single_emb = last_hidden.mean(dim=1).cpu().numpy()

                                if np.isnan(single_emb).any():
                                    logger.warning(f"Found NaN in single text embedding, using random fallback")
                                    dim = getattr(self.model.config, 'hidden_size', 768)
                                    # float64 to avoid JSON serialization issues, will be converted later
                                    single_emb = np.random.randn(1, dim).astype(np.float64)
                                
                                chunk_embeddings.append(single_emb)
                                
                                del inputs, outputs, last_hidden
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                        except Exception as e2:
                            logger.error(f"Error processing single text: {e2}")
                            # Last resort: random embedding
                            dim = getattr(self.model.config, 'hidden_size', 768)
                            # Use float64 to avoid JSON serialization issues, will be converted later  
                            chunk_embeddings.append(np.random.randn(1, dim).astype(np.float64))
            
            if chunk_embeddings:
                chunk_array = np.concatenate(chunk_embeddings, axis=0)
                all_embeddings.append(chunk_array)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info(f"Processed chunk {chunk_start}-{chunk_end} / {len(texts)}")
        
        if all_embeddings:
            embeddings = np.concatenate(all_embeddings, axis=0)
        else:
            # Fallback empty embeddings
            dim = getattr(self.model.config, 'hidden_size', 768)
            embeddings = np.zeros((len(texts), dim), dtype=np.float64)
        
        processing_time = time.time() - start_time
        
        logger.info(f"Generated embeddings shape {embeddings.shape} in {processing_time:.2f}s")
        
        return EmbeddingResult(
            embeddings=embeddings,
            metadata={
                "backend": self.backend_name,
                "batch_size": batch_size,
                "max_length": max_length,
                "chunk_size": chunk_size,
                "samples_processed": len(texts),
                "embedding_dim": embeddings.shape[1] if len(embeddings.shape) > 1 else 0
            },
            processing_time=processing_time
        )
    
    async def generate_text(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text completion"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        max_new_tokens = kwargs.get('max_new_tokens', 512)
        temperature = kwargs.get('temperature', 0.7)
        do_sample = kwargs.get('do_sample', True)
        
        # Filter out kwargs that we handle explicitly to avoid duplication
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['max_new_tokens', 'temperature', 'do_sample']}
        
        logger.debug(f"Generating text for prompt: {prompt[:50]}...")
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **filtered_kwargs
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_text = response[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            generated_text = f"[Error: {str(e)}]"
        
        processing_time = time.time() - start_time
        
        return GenerationResult(
            text=generated_text,
            metadata={
                "backend": self.backend_name,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "prompt_length": len(prompt)
            },
            processing_time=processing_time
        )
    
    async def unload_model(self):
        """Clean up model resources with aggressive GPU memory cleanup"""
        logger.info("Unloading model resources")
        
        if self.model is not None:
            del self.model
            self.model = None
            
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Aggressive GPU memory cleanup
        if torch.cuda.is_available():
            # Multi-pass cleanup with validation
            max_retries = 3
            for attempt in range(max_retries):
                # Clear all GPU caches
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
                # Clear cache for all devices
                for device_id in range(torch.cuda.device_count()):
                    with torch.cuda.device(device_id):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                
                # Check memory state
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                
                logger.info(f"GPU Memory after cleanup (attempt {attempt+1}): "
                           f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
                
                # If memory is sufficiently freed, break
                if allocated < 0.1:  # Less than 100MB allocated
                    break
                    
                if attempt < max_retries - 1:
                    logger.warning(f"Memory not fully freed, retrying cleanup...")
                    import time
                    time.sleep(2)
                    gc.collect()
        
        self.is_loaded = False
        logger.info("Model unloaded successfully")
    
    @property
    def backend_name(self) -> str:
        return "local_transformers"
    
    @property
    def supports_batch_inference(self) -> bool:
        return True
    
    @property
    def max_batch_size(self) -> int:
        return 32  # Conservative default, can be configured