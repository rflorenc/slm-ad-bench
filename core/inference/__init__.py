"""
Inference backend abstraction

Provides unified interface for different inference backends:
- Transformers (local)
- vLLM (tbd)
- Triton (tbd)
- Extensible to custom backends
"""

from .base import InferenceBackend, InferenceConfig, EmbeddingResult
from .factory import InferenceBackendFactory
from .manager import InferenceManager

__all__ = [
    'InferenceBackend',
    'InferenceConfig', 
    'EmbeddingResult',
    'InferenceBackendFactory',
    'InferenceManager'
]