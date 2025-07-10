import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .settings import BenchmarkConfig, ModelConfig, DatasetConfig, SystemConfig


class ConfigLoader:
    """Loads and manages YAML config files"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        if config_dir is None:
            config_dir = Path(__file__).parent
        self.config_dir = config_dir
    
    def load_yaml(self, filename: str) -> Dict[str, Any]:
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_models_config(self) -> Dict[str, ModelConfig]:
        data = self.load_yaml("models.yaml")
        configs = {}
        
        for model_data in data.get('models', []):
            configs[model_data['name']] = ModelConfig(
                name=model_data['name'],
                model_name=model_data['model_name'],
                supported_backends=model_data.get('supported_backends', ['local_transformers']),
                preferred_backend=model_data.get('preferred_backend', 'local_transformers'),
                backend_configs=model_data.get('backend_configs', {})
            )
        
        return configs
    
    def load_datasets_config(self) -> Dict[str, DatasetConfig]:
        data = self.load_yaml("datasets.yaml")
        configs = {}
        
        for dataset_data in data.get('datasets', []):
            configs[dataset_data['name']] = DatasetConfig(
                name=dataset_data['name'],
                path=dataset_data.get('path', ''),
                type=dataset_data.get('type', 'csv'),
                description=dataset_data.get('description', ''),
                preprocessing=dataset_data.get('preprocessing', {})
            )
        
        return configs
    
    def load_system_config(self) -> SystemConfig:
        try:
            data = self.load_yaml("system.yaml")
        except FileNotFoundError:
            data = {}
        
        return SystemConfig(
            device=data.get('device', 'auto'),
            memory_limit=data.get('memory_limit', 8192),
            max_workers=data.get('max_workers', 1),
            cache_dir=data.get('cache_dir', '.cache'),
            temp_dir=data.get('temp_dir', '/tmp'),
            logging_level=data.get('logging_level', 'INFO')
        )
    
    def load_benchmark_config(self) -> BenchmarkConfig:
        """Load full benchmark config"""
        models = self.load_models_config()
        datasets = self.load_datasets_config()
        system = self.load_system_config()
        
        return BenchmarkConfig(
            models=models,
            datasets=datasets,
            system=system
        )