"""
Evaluation Engine for SLM-AD-BENCH
Main evaluation orchestrator that integrates all evaluation types
"""

import numpy as np
import logging
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from ..inference.base import InferenceBackend
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.json_utils import ensure_json_serializable
from ..monitoring.comprehensive_monitor import monitor_operation
from ..monitoring import PlotManager, create_clustering_plots, save_anomaly_data
from .data_loading import (
    load_dataset, load_dataset_unlabeled, 
    estimate_contamination, enrich_text_for_rag
)
from .anomaly_detection import (
    detect_anomalies_unsupervised, detect_anomalies_clustering, evaluate_unsupervised_results
)

from .rag_system import RAGEvaluationSystem
from .reasoning_enhancement import ReasoningEnhancer
from .supervised import (
    run_unsw_nb15_traditional_evaluation, evaluate_llm_embeddings_cv
)

logger = logging.getLogger(__name__)

class EvaluationEngine:
    """
    Main evaluation engine that orchestrates different evaluation scenarios
    """
    
    def __init__(self, reasoning_config: Optional[Dict[str, Any]] = None):
        self.backend = None
        self.evaluation_stats = {}
        self.reasoning_enhancer = ReasoningEnhancer(reasoning_config or {}) if reasoning_config else None
        self.reasoning_config = reasoning_config or {}
        
    def _calculate_reasoning_sample_sizes(self, total_anomalies: int, total_normals: int) -> Tuple[int, int]:
        """
        Calculate how many anomaly and normal samples to analyze based on percentage configuration
        
        Args:
            total_anomalies: Total number of detected anomalies
            total_normals: Total number of normal samples
            
        Returns:
            Tuple of (num_anomaly_samples, num_normal_samples)
        """
        sampling_config = self.reasoning_config.get('sampling', {})
        
        # Get configuration values with defaults matching approaches.yaml
        anomaly_percentage = sampling_config.get('anomaly_percentage', 1.0)
        min_samples = sampling_config.get('min_samples', 3)
        max_samples = sampling_config.get('max_samples', 8)
        normal_samples = sampling_config.get('normal_samples', 2)
        
        # Calculate anomaly samples based on percentage (1.0 = 100%)
        anomaly_samples = max(1, int(total_anomalies * anomaly_percentage))
        
        # Apply min/max constraints
        anomaly_samples = max(min_samples, anomaly_samples)
        anomaly_samples = min(max_samples, anomaly_samples)
        
        # Don't exceed available samples
        anomaly_samples = min(anomaly_samples, total_anomalies)
        normal_samples = min(normal_samples, total_normals)
        
        logger.info(f"Calculated reasoning sample sizes: {anomaly_samples} anomalies ({anomaly_percentage*100:.1f}% of {total_anomalies}), {normal_samples} normals")
        
        return anomaly_samples, normal_samples
        
    def _load_dataset_config(self, data_path: str, dataset_type: str) -> Optional[Dict[str, Any]]:
        """Load dataset configuration from datasets.yaml"""
        try:
            import yaml
            from pathlib import Path
            
            config_path = Path(__file__).parent.parent.parent / "config" / "datasets.yaml"
            if not config_path.exists():
                logger.warning(f"Dataset config file not found: {config_path}")
                return None
                
            with open(config_path) as f:
                config_data = yaml.safe_load(f)
            
            abs_data_path = Path(data_path).resolve()
            for dataset_config in config_data.get('datasets', []):
                if dataset_config.get('dataset_type') == dataset_type:
                    config_data_path = Path(dataset_config.get('data_path', ''))
                    if config_data_path.is_absolute():
                        config_abs_path = config_data_path
                    else:
                        config_abs_path = (Path(__file__).parent.parent.parent / config_data_path).resolve()
                    
                    if config_abs_path == abs_data_path:
                        logger.info(f"Found matching dataset configuration: {dataset_config['name']}")
                        return dataset_config
                        
            # If no exact match, return first config with matching dataset_type
            for dataset_config in config_data.get('datasets', []):
                if dataset_config.get('dataset_type') == dataset_type:
                    logger.info(f"Using dataset configuration for {dataset_type}: {dataset_config['name']}")
                    return dataset_config
                    
            logger.warning(f"No dataset configuration found for {dataset_type}")
            return None
            
        except Exception as e:
            logger.error(f"Error loading dataset configuration: {e}")
            return None

    def save_embeddings(self, embeddings: np.ndarray, approach_name: str, results_dir: str):
        """Save embeddings as .npz file for compatibility with original system"""
        safe_name = approach_name.replace(" ", "_").replace("/", "_")
        embeddings_file = Path(results_dir) / f"embeddings_{safe_name}.npz"
        
        np.savez_compressed(embeddings_file, embeddings=embeddings)
        logger.info(f"Saved embeddings to {embeddings_file}")
        
    async def run_unsw_nb15_evaluation(self, backend: InferenceBackend, train_path: str, 
                                     test_path: str, nlines: int, approach_name: str, results_dir: str) -> List[Dict[str, Any]]:
        """
        Run UNSW-NB15 evaluation (labeled train/test split) with both traditional ML and LLM embeddings
        
        Args:
            backend: Inference backend to use
            train_path: Path to training data
            test_path: Path to test data  
            nlines: Number of lines to process
            approach_name: Name of the approach
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Running UNSW-NB15 evaluation: train={train_path}, test={test_path}, nlines={nlines}")
        
        results = []
        
        try:
            # 1. Traditional ML evaluation (baseline)
            logger.info("=== Traditional ML Evaluation ===")
            
            with monitor_operation("traditional_ml", Path(results_dir) / "monitoring" / approach_name) as monitor:
                traditional_results = run_unsw_nb15_traditional_evaluation(train_path, test_path, nlines)
                trad_summary = monitor._calculate_summary_stats(monitor.metrics_history, "traditional_ml") if monitor.metrics_history else {}
            
            for trad_result in traditional_results:
                trad_result.update({
                    "approach": f"{approach_name}_traditional_{trad_result['classification_type']}",
                    "classifier": "ExtraTrees_50_trees",
                    "eval_type": "research_compliant_traditional",
                    "methodology": "5_fold_cv_minmax_no_pca",
                    "backend": backend.backend_name,
                    "dataset_type": "unsw-nb15",
                    "nlines": nlines,
                    "cpu_usage_mean": trad_summary.get('cpu_percent_mean', 0),
                    "memory_usage_gb_mean": trad_summary.get('memory_used_gb_mean', 0),
                    "power_consumption_mean_w": trad_summary.get('total_estimated_power_mean', 0),
                    "duration_seconds": trad_summary.get('duration_seconds', 0),
                })
                results.append(trad_result)
            
            # 2. LLM Embeddings evaluation
            logger.info("=== LLM Embeddings Evaluation ===")
            
            # Load data as text for LLM processing (separate train/test like original)
            train_lines, y_train = load_dataset(train_path, "unsw-nb15", nlines)
            test_lines, y_test = load_dataset(test_path, "unsw-nb15", nlines)
            
            logger.info(f"Loaded train: {len(train_lines)} samples, test: {len(test_lines)} samples for LLM processing")
            
            # Generate embeddings separately for train and test
            with monitor_operation("embedding_generation", Path(results_dir) / "monitoring" / approach_name) as embed_monitor:
                train_embedding_result = await backend.generate_embeddings(
                    train_lines, 
                    batch_size=getattr(backend.config, 'batch_size', 2),
                    max_length=getattr(backend.config, 'max_length', 128)
                )
                train_embeddings = train_embedding_result.embeddings

                test_embedding_result = await backend.generate_embeddings(
                    test_lines,
                    batch_size=getattr(backend.config, 'batch_size', 2),
                    max_length=getattr(backend.config, 'max_length', 128)
                )
                test_embeddings = test_embedding_result.embeddings
                
                embed_summary = embed_monitor._calculate_summary_stats(embed_monitor.metrics_history, "embedding_generation") if embed_monitor.metrics_history else {}
            
            embeddings = np.vstack([train_embeddings, test_embeddings])
            all_labels = np.concatenate([y_train, y_test])
            total_time = train_embedding_result.processing_time + test_embedding_result.processing_time
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            self.save_embeddings(embeddings, approach_name, results_dir)
            
            # evaluate embeddings with classifiers with monitoring and PCA support
            with monitor_operation("classifier_evaluation", Path(results_dir) / "monitoring" / approach_name) as eval_monitor:
                # For UNSW-NB15: use original train/test split (don't re-split!)
                logger.info(f"Preserving original UNSW-NB15 train/test split: train={len(train_embeddings)}, test={len(test_embeddings)}")
                
                if train_embeddings.shape[1] > 768:
                    logger.info(f"Reducing from {train_embeddings.shape[1]} to 768 via pca...")
                    pca = PCA(n_components=768, random_state=42)
                    X_train_reduced = pca.fit_transform(train_embeddings)
                    X_test_reduced = pca.transform(test_embeddings)
                    pca_explained_var = sum(pca.explained_variance_ratio_)
                    logger.info(f"PCA explained variance: {pca_explained_var:.4f}")
                else:
                    X_train_reduced = train_embeddings
                    X_test_reduced = test_embeddings
                    pca_explained_var = 1.0000
                    logger.info(f"No PCA reduction needed (dim={train_embeddings.shape[1]})")
                
                plot_manager = PlotManager(results_dir)
                plot_manager.create_train_test_2d_plot(
                    X_train_reduced, y_train,
                    X_test_reduced, y_test,
                    approach_name, "UNSW-NB15", "LLM_train_vs_test"
                )
                
                # For evaluation: Use combined data for CV
                X_combined = np.vstack([X_train_reduced, X_test_reduced])
                y_combined = np.concatenate([y_train, y_test])
                llm_results = evaluate_llm_embeddings_cv(X_combined, y_combined, approach_name, "unsw-nb15")
                eval_summary = eval_monitor._calculate_summary_stats(eval_monitor.metrics_history, "classifier_evaluation") if eval_monitor.metrics_history else {}
            
            for llm_result in llm_results:
                llm_result.update({
                    "backend": backend.backend_name,
                    "dataset_type": "unsw-nb15", 
                    "nlines": int(nlines),
                    "embedding_time": float(total_time),
                    "sec_per_1k_logs": float(total_time / (nlines / 1000.0)) if nlines > 0 else 0.0,
                    "embedding_metadata": ensure_json_serializable({
                        "train_metadata": train_embedding_result.metadata,
                        "test_metadata": test_embedding_result.metadata
                    })
                })
                sanitized_result = ensure_json_serializable(llm_result)
                results.append(sanitized_result)

            logger.info("Starting RAG evaluation with training data only")
            
            # Skip RAG for vLLM backend to avoid memory issues
            if backend.backend_name == "vllm":
                logger.warning("Skipping RAG evaluation for vLLM backend due to memory constraints")
            else:
                rag_system = RAGEvaluationSystem()
                
                try:
                    # RAG for TRAIN data
                    rag_success = rag_system.setup_rag_database(
                        train_lines, backend.model, backend.tokenizer, 
                        reducer=pca if train_embeddings.shape[1] > 768 else None, embedding_type="llm"
                    )
                    
                    if rag_success:
                        # Run RAG on a subset of test data
                        test_subset = test_lines[:2]
                        test_indices = list(range(len(test_subset)))
                        
                        with monitor_operation("rag_explanation", Path(results_dir) / "monitoring" / approach_name) as rag_monitor:
                            rag_results = rag_system.evaluate_with_rag(
                                test_subset, test_indices, backend.model, backend.tokenizer,
                                labels=y_train, dataset_type="unsw-nb15", results_dir=results_dir
                            )

                            rag_system.save_rag_results(rag_results, approach_name, results_dir)
                            
                            rag_summary = rag_monitor._calculate_summary_stats(rag_monitor.metrics_history, "rag_explanation") if rag_monitor.metrics_history else {}
                            
                    else:
                        logger.warning("RAG database setup failed, skipping RAG evaluation")
                        
                except Exception as e:
                    logger.error(f"RAG evaluation failed: {e}")
                finally:
                    rag_system.cleanup()
                
            # Enhanced reasoning
            if self.reasoning_enhancer:
                logger.info("Starting enhanced reasoning evaluation for UNSW-NB15")
                
                # Get sample sizes based on actual labels
                anomaly_indices = [i for i, label in enumerate(y_test) if label == 1]
                normal_indices = [i for i, label in enumerate(y_test) if label == 0]
                
                anomaly_samples, normal_samples = self._calculate_reasoning_sample_sizes(
                    len(anomaly_indices), len(normal_indices)
                )
                
                selected_indices = []
                if len(anomaly_indices) > 0:
                    selected_indices.extend(anomaly_indices[:anomaly_samples])
                if len(normal_indices) > 0:
                    selected_indices.extend(normal_indices[:normal_samples])
                
                if len(selected_indices) == 0:
                    selected_indices = list(range(min(5, len(test_lines))))
                
                logger.info(f"Selected {len(selected_indices)} samples for reasoning: {len([i for i in selected_indices if i in anomaly_indices])} anomalies, {len([i for i in selected_indices if i in normal_indices])} normals")
                
                # Get selected samples
                selected_lines = [test_lines[i] for i in selected_indices]
                selected_labels = [y_test[i] for i in selected_indices]
                
                reasoning_results = await self._run_reasoning_enhancement_evaluation(
                    backend, selected_lines, selected_labels, "unsw-nb15", approach_name, results_dir
                )

                self._save_reasoning_results(reasoning_results, approach_name, results_dir)
            
            logger.info(f"UNSW-NB15 evaluation completed with {len(results)} results")
            
        except Exception as e:
            logger.error(f"UNSW-NB15 evaluation failed: {e}")
            results.append({
                "approach": f"{approach_name}_error",
                "error": str(e),
                "backend": backend.backend_name,
                "dataset_type": "unsw-nb15",
                "nlines": nlines
            })
        finally:
            if backend.is_loaded:
                await backend.unload_model()
        
        return results
    
    async def run_standard_evaluation(self, backend: InferenceBackend, data_path: str, 
                                    nlines: int, dataset_type: str, approach_name: str, results_dir: str) -> List[Dict[str, Any]]:
        """
        Run standard evaluation for single-file datasets (EventTraces, etc.)
        
        Args:
            backend: Inference backend to use
            data_path: Path to dataset file
            nlines: Number of lines to process  
            dataset_type: Type of dataset
            approach_name: Name of the approach
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Running standard evaluation: {dataset_type}, path={data_path}, nlines={nlines}")
        
        results = []
        
        try:
            # Load dataset configuration from datasets.yaml
            dataset_config = self._load_dataset_config(data_path, dataset_type)
            preprocessing = dataset_config.get('preprocessing', {}) if dataset_config else {}
            
            lines, labels = load_dataset(data_path, dataset_type, nlines, preprocessing)
            logger.info(f"Loaded {len(lines)} samples for {dataset_type} dataset")
            
            # split data first 80/20 split before embedding generation)
            train_lines, test_lines, y_train, y_test = train_test_split(
                lines, labels, test_size=0.2, random_state=42, stratify=labels
            )
            
            logger.info(f"Train set: {len(train_lines)} samples, Test set: {len(test_lines)} samples")
            
            # generate embeddings with monitoring
            with monitor_operation("embedding_generation", Path(results_dir) / "monitoring" / approach_name) as embed_monitor:
                # 1: Gen train embeddings
                train_embedding_result = await backend.generate_embeddings(
                    train_lines,
                    batch_size=getattr(backend.config, 'batch_size', 2),
                    max_length=getattr(backend.config, 'max_length', 128)
                )
                train_embeddings = train_embedding_result.embeddings
                logger.info(f"Created embeddings with shape {train_embeddings.shape}")
                
                # 2: Gen test embeddings  
                test_embedding_result = await backend.generate_embeddings(
                    test_lines,
                    batch_size=getattr(backend.config, 'batch_size', 2),
                    max_length=getattr(backend.config, 'max_length', 128)
                )
                test_embeddings = test_embedding_result.embeddings
                logger.info(f"Created embeddings with shape {test_embeddings.shape}")
                
                embed_summary = embed_monitor._calculate_summary_stats(embed_monitor.metrics_history, "embedding_generation") if embed_monitor.metrics_history else {}
            
            embeddings = np.vstack([train_embeddings, test_embeddings])
            all_labels = np.concatenate([y_train, y_test])
            total_time = train_embedding_result.processing_time + test_embedding_result.processing_time
            
            logger.info(f"Combined embeddings shape: {embeddings.shape}")

            self.save_embeddings(embeddings, approach_name, results_dir)
            
            with monitor_operation("classifier_evaluation", Path(results_dir) / "monitoring" / approach_name) as eval_monitor:
                # EventTraces: Preserve the initial train/test split (don't re-split!)
                logger.info(f"Preserving initial {dataset_type} train/test split: train={len(train_embeddings)}, test={len(test_embeddings)}")
                
                if train_embeddings.shape[1] > 768:
                    max_components = min(768, train_embeddings.shape[0] - 1, train_embeddings.shape[1])
                    logger.info(f"Reducing from {train_embeddings.shape[1]} to {max_components} via pca...")
                    pca = PCA(n_components=max_components, random_state=42)
                    X_train_reduced = pca.fit_transform(train_embeddings)
                    X_test_reduced = pca.transform(test_embeddings)
                    pca_explained_var = sum(pca.explained_variance_ratio_)
                    logger.info(f"PCA explained variance: {pca_explained_var:.4f}")
                else:
                    X_train_reduced = train_embeddings
                    X_test_reduced = test_embeddings
                    pca_explained_var = 1.0000
                    logger.info(f"No PCA reduction needed (dim={train_embeddings.shape[1]})")
                
                # Create proper train/test plots using original split
                plot_manager = PlotManager(results_dir)
                plot_manager.create_train_test_2d_plot(
                    X_train_reduced, y_train,
                    X_test_reduced, y_test,
                    approach_name, dataset_type, "train_vs_test"
                )
                
                # For evaluation: Use combined data for CV (matching original)
                X_combined = np.vstack([X_train_reduced, X_test_reduced])
                y_combined = np.concatenate([y_train, y_test])
                llm_results = evaluate_llm_embeddings_cv(X_combined, y_combined, approach_name, dataset_type)
                eval_summary = eval_monitor._calculate_summary_stats(eval_monitor.metrics_history, "classifier_evaluation") if eval_monitor.metrics_history else {}
            
            for llm_result in llm_results:
                llm_result.update({
                    "backend": backend.backend_name,
                    "dataset_type": dataset_type,
                    "nlines": int(nlines),
                    "embedding_time": float(total_time),
                    "sec_per_1k_logs": float(total_time / (nlines / 1000.0)) if nlines > 0 else 0.0,
                    "embedding_metadata": ensure_json_serializable({
                        "train_metadata": train_embedding_result.metadata,
                        "test_metadata": test_embedding_result.metadata
                    })
                })

                sanitized_result = ensure_json_serializable(llm_result)
                results.append(sanitized_result)


            logger.info("Starting RAG evaluation with training data only")
            
            # Skip RAG for vLLM backend to avoid memory issues
            if backend.backend_name == "vllm":
                logger.warning("Skipping RAG evaluation for vLLM backend due to memory constraints")
            else:
                rag_system = RAGEvaluationSystem()
                
                try:
                    rag_success = rag_system.setup_rag_database(
                        train_lines, backend.model, backend.tokenizer, 
                        reducer=pca if train_embeddings.shape[1] > 768 else None, embedding_type="llm"
                    )
                    
                    if rag_success:
                        test_subset = test_lines[:2]
                        test_indices = list(range(len(test_subset)))
                        
                        with monitor_operation("rag_explanation", Path(results_dir) / "monitoring" / approach_name) as rag_monitor:
                            rag_results = rag_system.evaluate_with_rag(
                                test_subset, test_indices, backend.model, backend.tokenizer,
                                labels=y_train, dataset_type=dataset_type, results_dir=results_dir
                            )
                            
                            rag_system.save_rag_results(rag_results, approach_name, results_dir)
                            rag_summary = rag_monitor._calculate_summary_stats(rag_monitor.metrics_history, "rag_explanation") if rag_monitor.metrics_history else {}
                            
                    else:
                        logger.warning("RAG database setup failed, skipping RAG evaluation")
                        
                except Exception as e:
                    logger.error(f"RAG evaluation failed: {e}")
                finally:
                    rag_system.cleanup()
                
            # Enhanced Reasoning 
            if self.reasoning_enhancer:
                logger.info("Starting enhanced reasoning evaluation")

                anomaly_indices = [i for i, label in enumerate(y_test) if label == 1]
                normal_indices = [i for i, label in enumerate(y_test) if label == 0]
                
                anomaly_samples, normal_samples = self._calculate_reasoning_sample_sizes(
                    len(anomaly_indices), len(normal_indices)
                )

                selected_indices = []
                if len(anomaly_indices) > 0:
                    selected_indices.extend(anomaly_indices[:anomaly_samples])
                if len(normal_indices) > 0:
                    selected_indices.extend(normal_indices[:normal_samples])

                if len(selected_indices) == 0:
                    selected_indices = list(range(min(5, len(test_lines))))
                
                logger.info(f"Selected {len(selected_indices)} samples for reasoning: {len([i for i in selected_indices if i in anomaly_indices])} anomalies, {len([i for i in selected_indices if i in normal_indices])} normals")
                
                # Get selected samples
                selected_lines = [test_lines[i] for i in selected_indices]
                selected_labels = [y_test[i] for i in selected_indices]
                
                reasoning_results = await self._run_reasoning_enhancement_evaluation(
                    backend, selected_lines, selected_labels, dataset_type, approach_name, results_dir
                )
                self._save_reasoning_results(reasoning_results, approach_name, results_dir)
            
            logger.info(f"Standard evaluation completed with {len(results)} results")
            
        except Exception as e:
            logger.error(f"Standard evaluation failed: {e}")
            results.append({
                "approach": f"{approach_name}_error",
                "error": str(e),
                "backend": backend.backend_name,
                "dataset_type": dataset_type,
                "nlines": nlines
            })
        finally:
            if backend.is_loaded:
                await backend.unload_model()
        
        return results
    
    async def _run_reasoning_enhancement_evaluation(self, backend: InferenceBackend, 
                                                   test_lines: List[str], test_labels: List[int],
                                                   dataset_type: str, approach_name: str, 
                                                   results_dir: str) -> List[Dict[str, Any]]:
        """Run reasoning enhancement evaluation on test samples"""
        
        reasoning_results = []
        
        for i, (log_entry, true_label) in enumerate(zip(test_lines, test_labels)):
            logger.info(f"Processing reasoning enhancement for sample {i+1}/{len(test_lines)}")
            
            try:
                # Apply reasoning enhancements
                enhanced_result = await self.reasoning_enhancer.enhance_prediction(
                    backend, log_entry, dataset_type
                )
                
                # Add evaluation metadata
                enhanced_result.update({
                    'sample_index': i,
                    'true_label': true_label,
                    'approach': approach_name,
                    'dataset_type': dataset_type,
                    'evaluation_type': 'reasoning_enhancement'
                })
                
                reasoning_results.append(enhanced_result)
                
            except Exception as e:
                logger.error(f"Reasoning enhancement failed for sample {i}: {e}")
                reasoning_results.append({
                    'sample_index': i,
                    'true_label': true_label,
                    'approach': approach_name,
                    'dataset_type': dataset_type,
                    'evaluation_type': 'reasoning_enhancement',
                    'error': str(e)
                })
        
        return reasoning_results
    
    def _save_reasoning_results(self, reasoning_results: List[Dict[str, Any]], 
                               approach_name: str, results_dir: str):
        """Save reasoning enhancement results to JSON file"""
        
        if not reasoning_results:
            return
            
        results_dir = Path(results_dir)
        safe_name = approach_name.replace(" ", "_").replace("/", "_")
        reasoning_file = results_dir / f"reasoning_results_{safe_name}.json"

        sanitized_results = ensure_json_serializable(reasoning_results)
        
        with open(reasoning_file, 'w') as f:
            json.dump(sanitized_results, f, indent=2)
        
        logger.info(f"Saved reasoning enhancement results to {reasoning_file}")
    
    async def _run_reasoning_enhancement_evaluation_unlabeled(self, backend: InferenceBackend, 
                                                           sample_lines: List[str],
                                                           dataset_type: str, approach_name: str, 
                                                           results_dir: str, sample_indices: List[int] = None,
                                                           predictions: np.ndarray = None) -> List[Dict[str, Any]]:
        """Run reasoning enhancement evaluation on unlabeled samples (no true labels)"""
        
        reasoning_results = []
        
        for i, log_entry in enumerate(sample_lines):
            actual_index = sample_indices[i] if sample_indices else i
            detected_as_anomaly = predictions[actual_index] == 1 if predictions is not None else None
            anomaly_status = "DETECTED_ANOMALY" if detected_as_anomaly else "DETECTED_NORMAL" if detected_as_anomaly is not None else "UNKNOWN"
            
            logger.info(f"Processing reasoning enhancement for unlabeled sample {i+1}/{len(sample_lines)} (index {actual_index}, {anomaly_status})")
            
            try:
                # Apply reasoning enhancements (same as labeled, but without true labels)
                enhanced_result = await self.reasoning_enhancer.enhance_prediction(
                    backend, log_entry, dataset_type
                )
                
                # Add evaluation metadata (no true label available, but include detection status)
                enhanced_result.update({
                    'sample_index': actual_index,
                    'true_label': None,  # No true label for unlabeled data
                    'detected_as_anomaly': detected_as_anomaly,
                    'anomaly_status': anomaly_status,
                    'approach': approach_name,
                    'dataset_type': dataset_type,
                    'evaluation_type': 'reasoning_enhancement_unlabeled'
                })
                
                reasoning_results.append(enhanced_result)
                
            except Exception as e:
                logger.error(f"Reasoning enhancement failed for unlabeled sample {i}: {e}")
                reasoning_results.append({
                    'sample_index': actual_index,
                    'true_label': None,
                    'detected_as_anomaly': detected_as_anomaly,
                    'anomaly_status': anomaly_status,
                    'approach': approach_name,
                    'dataset_type': dataset_type,
                    'evaluation_type': 'reasoning_enhancement_unlabeled',
                    'error': str(e)
                })
        
        return reasoning_results
    
    async def run_unlabeled_evaluation(self, backend: InferenceBackend, data_path: str, 
                                     nlines: int, dataset_type: str, contamination: str,
                                     approach_name: str, results_dir: str) -> List[Dict[str, Any]]:
        """
        Run unlabeled/unsupervised evaluation
        
        Args:
            backend: Inference backend to use  
            data_path: Path to dataset file
            nlines: Number of lines to process
            dataset_type: Type of dataset
            contamination: Contamination parameter ('auto' or float)
            approach_name: Name of the approach
            
        Returns:
            List of evaluation results
        """
        logger.info(f"Running unlabeled evaluation: {dataset_type}, path={data_path}, nlines={nlines}, contamination={contamination}")
        
        results = []
        
        try:
            # Load unlabeled data
            if dataset_type == "unsw-nb15":
                X_scaled, _, lines = load_dataset_unlabeled(data_path, "unsw-nb15", nlines)
                # Use the text representation for LLM processing
                data_for_llm = lines
            else:
                # For HDFS/EventTraces
                lines, _, metadata = load_dataset_unlabeled(data_path, "hdfs", nlines)
                data_for_llm = lines
            
            logger.info(f"Loaded {len(data_for_llm)} unlabeled samples")
            
            # Generate embeddings using LLM with monitoring
            with monitor_operation("embedding_generation", Path(results_dir) / "monitoring" / approach_name) as embed_monitor:
                embedding_result = await backend.generate_embeddings(
                    data_for_llm,
                    batch_size=getattr(backend.config, 'batch_size', 2),
                    max_length=getattr(backend.config, 'max_length', 128)
                )
                embed_summary = embed_monitor._calculate_summary_stats(embed_monitor.metrics_history, "embedding_generation") if embed_monitor.metrics_history else {}
            
            embeddings = embedding_result.embeddings
            logger.info(f"Generated embeddings shape: {embeddings.shape}")
            
            # Save embeddings for compatibility
            self.save_embeddings(embeddings, approach_name, results_dir)
            
            if contamination == "auto":
                contamination_val = estimate_contamination(embeddings, method='statistical')
            else:
                try:
                    contamination_val = float(contamination)
                    contamination_val = max(0.001, min(0.5, contamination_val))
                except ValueError:
                    logger.warning(f"Invalid contamination value: {contamination}, using auto")
                    contamination_val = estimate_contamination(embeddings, method='statistical')
            
            logger.info(f"Using contamination rate: {contamination_val:.3f}")
            
            # Run unsupervised ad with monitoring
            with monitor_operation("anomaly_detection", Path(results_dir) / "monitoring" / approach_name) as anomaly_monitor:
                anomaly_results = detect_anomalies_unsupervised(
                    embeddings, 
                    contamination=contamination_val, 
                    method='ensemble'
                )
                
                # run clustering-based detection for plots
                clustering_results = detect_anomalies_clustering(
                    embeddings,
                    contamination=contamination_val,
                    approach_name=approach_name
                )
                anomaly_summary = anomaly_monitor._calculate_summary_stats(anomaly_monitor.metrics_history, "anomaly_detection") if anomaly_monitor.metrics_history else {}
            
            # Get ensemble predictions and scores
            if 'ensemble' in anomaly_results:
                predictions = anomaly_results['ensemble']
                scores = anomaly_results.get('ensemble_scores', predictions.astype(float))
            else:
                method_name = next(iter(anomaly_results['individual_predictions'].keys()))
                predictions = anomaly_results['individual_predictions'][method_name]
                scores = anomaly_results['scores'][method_name]

            create_clustering_plots(clustering_results, approach_name, results_dir)

            save_anomaly_data(predictions, scores, embeddings, approach_name, results_dir)

            # Extract cluster labels from clustering results for quality metrics
            clustering_labels = None
            if clustering_results:
                if 'hdbscan' in clustering_results:
                    clustering_labels = clustering_results['hdbscan']['labels']
                elif 'dbscan' in clustering_results:
                    clustering_labels = clustering_results['dbscan']['labels']
            
            eval_metrics = evaluate_unsupervised_results(
                embeddings, predictions, clustering_labels=clustering_labels, anomaly_scores=scores
            )

            result = {
                "approach": f"{approach_name}_unsupervised_llm",
                "dataset_type": dataset_type,
                "contamination": float(contamination_val),
                "contamination_input": contamination,
                "anomalies_detected": int(eval_metrics['num_anomalies']),
                "total_samples": int(eval_metrics['num_samples']),
                "anomaly_ratio": float(eval_metrics['anomaly_ratio']),
                "backend": backend.backend_name,
                "mode": "unlabeled",
                "nlines": int(nlines),
                "embedding_time": float(embedding_result.processing_time),
                "embedding_dim": int(embeddings.shape[1]),
                "embedding_metadata": ensure_json_serializable(embedding_result.metadata),
                
                "score_mean": float(eval_metrics.get('score_mean', 0.0)) if not np.isnan(eval_metrics.get('score_mean', 0.0)) else 0.0,
                "score_std": float(eval_metrics.get('score_std', 0.0)) if not np.isnan(eval_metrics.get('score_std', 0.0)) else 0.0,
                "score_separation": float(eval_metrics.get('score_separation', 0.0)) if not np.isnan(eval_metrics.get('score_separation', 0.0)) else 0.0,

                "num_methods_used": len(anomaly_results.get('individual_predictions', {})),
                "silhouette_score": float(eval_metrics.get('silhouette_score', 0.0)) if not np.isnan(eval_metrics.get('silhouette_score', 0.0)) else 0.0,
                "davies_bouldin_score": float(eval_metrics.get('davies_bouldin_score', 0.0)) if not np.isnan(eval_metrics.get('davies_bouldin_score', 0.0)) else 0.0
            }
            

            sanitized_result = ensure_json_serializable(result)
            results.append(sanitized_result)
            
            logger.info(f"Unlabeled evaluation completed: {result['anomalies_detected']}/{result['total_samples']} "
                       f"anomalies ({result['anomaly_ratio']:.3f} ratio)")
            
            # Enhanced Reasoning for unlabeled data
            if self.reasoning_enhancer:
                logger.info("Starting enhanced reasoning evaluation for unlabeled data")
                # Use actual detected anomalies for reasoning analysis
                anomaly_indices = np.where(predictions == 1)[0]  # Get anomaly indices (1 = anomaly)
                normal_indices = np.where(predictions == 0)[0]   # Get normal indices (0 = normal)
                
                # Calculate sample sizes based on detected anomalies/normals
                anomaly_samples, normal_samples = self._calculate_reasoning_sample_sizes(
                    len(anomaly_indices), len(normal_indices)
                )
                
                selected_indices = []
                if len(anomaly_indices) > 0:
                    selected_indices.extend(anomaly_indices[:anomaly_samples])
                if len(normal_indices) > 0:
                    selected_indices.extend(normal_indices[:normal_samples])
                
                if len(selected_indices) == 0:
                    selected_indices = list(range(min(5, len(lines))))
                
                sample_lines = [lines[i] for i in selected_indices]
                logger.info(f"Selected {len(sample_lines)} samples for reasoning: {len(anomaly_indices)} anomalies available, analyzing {len([i for i in selected_indices if i in anomaly_indices])} anomalies and {len([i for i in selected_indices if i in normal_indices])} normals at indices {selected_indices}")
                
                reasoning_results = await self._run_reasoning_enhancement_evaluation_unlabeled(
                    backend, sample_lines, dataset_type, approach_name, results_dir, selected_indices, predictions
                )
                
                # Save enhanced reasoning results
                self._save_reasoning_results(reasoning_results, approach_name, results_dir)
            
        except Exception as e:
            logger.error(f"Unlabeled evaluation failed: {e}")
            result = {
                "approach": f"{approach_name}_unsupervised_error",
                "dataset_type": dataset_type,
                "error": str(e),
                "backend": backend.backend_name,
                "mode": "unlabeled",
                "nlines": nlines
            }
            results.append(result)
        finally:
            if backend.is_loaded:
                await backend.unload_model()
        
        return results
    
    def get_evaluation_stats(self) -> Dict[str, Any]:
        return self.evaluation_stats.copy()