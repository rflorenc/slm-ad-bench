"""
Inference manager for backend lifecycle and fallback handling
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any, List, Union

from .base import InferenceBackend, InferenceConfig, EmbeddingResult, GenerationResult
from .factory import InferenceBackendFactory

logger = logging.getLogger(__name__)

class InferenceManager:
    """Manages inference backend lifecycle and provides fallback mechanisms"""
    
    def __init__(self):
        self.current_backend: Optional[InferenceBackend] = None
        self.backend_stats: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
    
    @asynccontextmanager
    async def backend_context(self, backend_type: str, config: Union[Dict, InferenceConfig]):
        """Context manager for backend lifecycle"""
        backend = None
        try:
            async with self._lock:
                logger.info(f"Initializing {backend_type} backend...")
                backend = InferenceBackendFactory.create_backend(backend_type, config)
                
                logger.info(f"Loading {backend_type} model...")
                load_stats = await backend.load_model()
                self.backend_stats[backend_type] = load_stats
                self.current_backend = backend
                
                logger.info(f"Backend {backend_type} ready (load time: {load_stats.get('load_time', 0):.2f}s)")
            
            yield backend
            
        except Exception as e:
            logger.error(f"Error with backend {backend_type}: {e}")
            raise
        finally:
            if backend and backend.is_loaded:
                try:
                    logger.info(f"Unloading {backend_type} backend...")
                    await backend.unload_model()
                except Exception as e:
                    logger.error(f"Error unloading backend {backend_type}: {e}")
            
            async with self._lock:
                self.current_backend = None
    
    async def generate_embeddings_with_fallback(self, 
                                              texts: List[str], 
                                              primary_backend: str, 
                                              primary_config: Union[Dict, InferenceConfig],
                                              fallback_backends: Optional[List[tuple]] = None,
                                              **kwargs) -> EmbeddingResult:
        """
        Generate embeddings with fallback support
        
        Args:
            texts: Input texts to embed
            primary_backend: Primary backend to try
            primary_config: Configuration for primary backend
            fallback_backends: List of (backend_type, config) tuples for fallback
            **kwargs: Additional arguments for embedding generation
        """
        backends_to_try = [(primary_backend, primary_config)]
        if fallback_backends:
            backends_to_try.extend(fallback_backends)
        
        last_error = None
        
        for i, (backend_type, config) in enumerate(backends_to_try):
            try:
                logger.info(f"Attempting embedding generation with {backend_type} " +
                           f"({'primary' if i == 0 else f'fallback {i}'})")
                
                async with self.backend_context(backend_type, config) as backend:
                    result = await backend.generate_embeddings(texts, **kwargs)
                    
                    result.metadata["fallback_used"] = i > 0
                    result.metadata["backend_attempt"] = i + 1
                    
                    logger.info(f"Successfully generated embeddings using {backend_type}")
                    return result
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Backend {backend_type} failed: {e}")
                
                if i == len(backends_to_try) - 1:
                    logger.error(f"All backends failed. Last error: {e}")
                    raise
                
                logger.info(f"Trying next fallback backend...")
                continue
        
        # should never be reached, but ...
        raise RuntimeError(f"All backends failed. Last error: {last_error}")
    
    async def generate_text_with_fallback(self, 
                                        prompt: str, 
                                        primary_backend: str, 
                                        primary_config: Union[Dict, InferenceConfig],
                                        fallback_backends: Optional[List[tuple]] = None,
                                        **kwargs) -> GenerationResult:
        """
        Generate text with fallback support
        
        Args:
            prompt: Input prompt
            primary_backend: Primary backend to try
            primary_config: Configuration for primary backend
            fallback_backends: List of (backend_type, config) tuples for fallback
            **kwargs: Additional arguments for text generation
        """
        backends_to_try = [(primary_backend, primary_config)]
        if fallback_backends:
            backends_to_try.extend(fallback_backends)
        
        last_error = None
        
        for i, (backend_type, config) in enumerate(backends_to_try):
            try:
                logger.info(f"Attempting text generation with {backend_type} " +
                           f"({'primary' if i == 0 else f'fallback {i}'})")
                
                async with self.backend_context(backend_type, config) as backend:
                    result = await backend.generate_text(prompt, **kwargs)
                    
                    result.metadata["fallback_used"] = i > 0
                    result.metadata["backend_attempt"] = i + 1
                    
                    logger.info(f"Successfully generated text using {backend_type}")
                    return result
                    
            except Exception as e:
                last_error = e
                logger.warning(f"Backend {backend_type} failed: {e}")
                
                if i == len(backends_to_try) - 1:
                    logger.error(f"All backends failed. Last error: {e}")
                    raise
                
                logger.info(f"Trying next fallback backend...")
                continue

        raise RuntimeError(f"All backends failed. Last error: {last_error}")
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get statistics for all backends used in this session"""
        return self.backend_stats.copy()
    
    def get_current_backend_info(self) -> Optional[Dict[str, Any]]:
        """Get information about currently loaded backend"""
        if self.current_backend is None:
            return None
        
        return {
            "backend_name": self.current_backend.backend_name,
            "is_loaded": self.current_backend.is_loaded,
            "supports_batch": self.current_backend.supports_batch_inference,
            "max_batch_size": self.current_backend.max_batch_size,
            "config": self.current_backend.config.__dict__
        }