"""
Base classes for inference backend abstraction
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class InferenceConfig:
    """Base configuration for inference backends"""
    model_name: str
    batch_size: int = 2
    max_length: int = 128
    device: str = "cuda"
    dtype: str = "float16"
    
@dataclass
class EmbeddingResult:
    """Standardized embedding result"""
    embeddings: np.ndarray
    metadata: Dict[str, Any]
    processing_time: float
    
@dataclass
class GenerationResult:
    """Standardized text generation result"""
    text: str
    metadata: Dict[str, Any]
    processing_time: float

class InferenceBackend(ABC):
    """Abstract base class for inference backends"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self._loading_stats = {}
    
    @abstractmethod
    async def load_model(self) -> Dict[str, Any]:
        """Load model and return loading stats"""
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str], **kwargs) -> EmbeddingResult:
        """Generate embeddings for input texts"""
        pass
    
    @abstractmethod
    async def generate_text(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate text completion"""
        pass
    
    @abstractmethod
    async def unload_model(self):
        """Clean up model resources"""
        pass
    
    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return backend identifier"""
        pass
    
    @property
    @abstractmethod
    def supports_batch_inference(self) -> bool:
        """Whether backend supports batch inference"""
        pass
    
    @property
    @abstractmethod
    def max_batch_size(self) -> int:
        """Maximum supported batch size"""
        pass
    
    @property
    def loading_stats(self) -> Dict[str, Any]:
        """Return model loading statistics"""
        return self._loading_stats.copy()
    
    def __enter__(self):
        """Sync context manager entry"""
        import asyncio
        loop = asyncio.get_event_loop()
        self._loading_stats = loop.run_until_complete(self.load_model())
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Sync context manager exit"""
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.unload_model())
    
    async def __aenter__(self):
        """Async context manager entry"""
        self._loading_stats = await self.load_model()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.unload_model()