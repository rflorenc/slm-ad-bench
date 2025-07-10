"""
Factory for creating inference backends
"""

from typing import Dict, Type, Union, List
import logging

from .base import InferenceBackend, InferenceConfig
from .local_transformers import LocalTransformersBackend, LocalTransformersConfig

logger = logging.getLogger(__name__)

class InferenceBackendFactory:

    _backends: Dict[str, Type[InferenceBackend]] = {
        "local_transformers": LocalTransformersBackend,
    }
    
    _configs: Dict[str, Type[InferenceConfig]] = {
        "local_transformers": LocalTransformersConfig,
    }
    
    @classmethod
    def create_backend(cls, backend_type: str, config: Union[Dict, InferenceConfig]) -> InferenceBackend:
        """Create inference backend instance"""
        if backend_type not in cls._backends:
            available = ", ".join(cls._backends.keys())
            raise ValueError(f"Unknown backend type: {backend_type}. Available: {available}")
        
        backend_class = cls._backends[backend_type]
        config_class = cls._configs[backend_type]
        
        # Convert dict to config object if needed
        if isinstance(config, dict):
            try:
                config = config_class(**config)
            except TypeError as e:
                raise ValueError(f"Invalid configuration for backend {backend_type}: {e}")
        elif not isinstance(config, config_class):
            raise ValueError(f"Config must be dict or {config_class.__name__} for backend {backend_type}")
        
        logger.info(f"Creating {backend_type} backend with model {config.model_name}")
        return backend_class(config)
    
    @classmethod
    def register_backend(cls, name: str, backend_class: Type[InferenceBackend], config_class: Type[InferenceConfig]):
        """Register new backend type"""
        if not issubclass(backend_class, InferenceBackend):
            raise ValueError("backend_class must be a subclass of InferenceBackend")
        if not issubclass(config_class, InferenceConfig):
            raise ValueError("config_class must be a subclass of InferenceConfig")
        
        cls._backends[name] = backend_class
        cls._configs[name] = config_class
        logger.info(f"Registered backend: {name}")
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List available backend types"""
        return list(cls._backends.keys())
    
    @classmethod
    def get_backend_info(cls, backend_type: str) -> Dict[str, Type]:
        """Get backend and config classes for a backend type"""
        if backend_type not in cls._backends:
            raise ValueError(f"Unknown backend type: {backend_type}")
        
        return {
            "backend_class": cls._backends[backend_type],
            "config_class": cls._configs[backend_type]
        }