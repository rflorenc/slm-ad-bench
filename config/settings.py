"""
Configuration classes
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from pathlib import Path

@dataclass
class ModelConfig:
    """Model specific config"""
    name: str
    model_name: str
    supported_backends: List[str] = field(default_factory=lambda: ["local_transformers"])
    preferred_backend: str = "local_transformers"
    backend_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def get_backend_config(self, backend_type: str) -> Dict[str, Any]:
        base_config = {
            "model_name": self.model_name,
            "batch_size": 2,
            "max_length": 128
        }
        
        if backend_type in self.backend_configs:
            base_config.update(self.backend_configs[backend_type])
        
        return base_config

@dataclass
class DatasetConfig:
    """Config for dataset loading and processing"""
    name: str
    dataset_type: str  # "unsw-nb15", "eventtraces", etc.
    train_path: Optional[str] = None
    test_path: Optional[str] = None
    data_path: Optional[str] = None
    nlines: Optional[int] = None
    contamination: float = 0.1
    preprocessing: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self):
        if self.dataset_type == "unsw-nb15":
            if not (self.train_path and self.test_path):
                raise ValueError("UNSW-NB15 requires both train_path and test_path")
        else:
            if not self.data_path:
                raise ValueError(f"Dataset type {self.dataset_type} requires data_path")

@dataclass
class SystemConfig:
    output_dir: str = "output_results"
    sample_interval: float = 0.5
    save_csv: bool = True
    create_plots: bool = True
    enable_monitoring: bool = True
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    @classmethod
    def from_env(cls) -> 'SystemConfig':
        return cls(
            output_dir=os.getenv('BENCHMARK_OUTPUT_DIR', 'output_results'),
            sample_interval=float(os.getenv('BENCHMARK_SAMPLE_INTERVAL', '0.5')),
            save_csv=os.getenv('BENCHMARK_SAVE_CSV', 'true').lower() == 'true',
            create_plots=os.getenv('BENCHMARK_CREATE_PLOTS', 'true').lower() == 'true',
            enable_monitoring=os.getenv('BENCHMARK_ENABLE_MONITORING', 'true').lower() == 'true',
            log_level=os.getenv('BENCHMARK_LOG_LEVEL', 'INFO'),
            log_file=os.getenv('BENCHMARK_LOG_FILE')
        )
    
    def create_output_dir(self) -> Path:
        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path

@dataclass
class InferenceConfig:
    default_backend: str = "local_transformers"
    fallback_backends: List[str] = field(default_factory=list)
    backend_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    timeout_seconds: Optional[int] = None
    retry_attempts: int = 1
    
    def get_backend_config(self, backend_type: str) -> Dict[str, Any]:
        return self.backend_configs.get(backend_type, {})

@dataclass
class EvaluationConfig:
    cv_folds: int = 5
    test_size: float = 0.2
    random_state: int = 42
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "roc_auc"])
    classifiers: List[str] = field(default_factory=lambda: ["RandomForest", "ExtraTrees", "LogisticRegression"])
    enable_rag: bool = True
    rag_top_k: int = 3
    enable_traditional_ml: bool = True
    
@dataclass
class BenchmarkConfig:
    """Main benchmark configs"""
    models: List[ModelConfig] = field(default_factory=list)
    datasets: List[DatasetConfig] = field(default_factory=list)
    system: SystemConfig = field(default_factory=SystemConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        for model in self.models:
            if model.name == model_name:
                return model
        return None
    
    def get_dataset_config(self, dataset_name: str) -> Optional[DatasetConfig]:
        for dataset in self.datasets:
            if dataset.name == dataset_name:
                return dataset
        return None
    
    def validate(self):
        """Validate entire configs"""
        if not self.models:
            raise ValueError("At least one model must be configured")
        
        if not self.datasets:
            raise ValueError("At least one dataset must be configured")
        
        for dataset in self.datasets:
            dataset.validate()
        
        for model in self.models:
            if model.preferred_backend not in model.supported_backends:
                raise ValueError(f"Model {model.name} preferred backend not in supported backends")