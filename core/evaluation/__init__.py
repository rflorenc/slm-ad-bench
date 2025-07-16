"""
Evaluation module for SLM-AD-BENCH
"""

from .data_loading import (
    load_dataset, load_dataset_unlabeled, 
    load_event_traces, load_unsw_nb15, load_unsw_nb15_cleaned,
    load_hdfs_unlabeled, load_unsw_nb15_unlabeled,
    estimate_contamination
)

from .anomaly_detection import (
    detect_anomalies_unsupervised, detect_anomalies_clustering,
    evaluate_unsupervised_results
)

from .advanced_evaluation import (
    reduce_dimensionality, split_embeddings_and_labels,
    plot_train_test_2d, create_comprehensive_evaluation_split,
    prepare_evaluation_data
)

from .rag_system import (
    RAGEvaluationSystem, store_embeddings_in_milvus,
    rag_explanation_monitored, cleanup_milvus_db
)

from .supervised import (
    evaluate_classifiers_cv, evaluate_extratrees_cv,
    run_unsw_nb15_traditional_evaluation, evaluate_llm_embeddings_cv
)

from .engine import EvaluationEngine

__all__ = [
    # Data loading
    'load_dataset', 'load_dataset_unlabeled',
    'load_event_traces', 'load_unsw_nb15', 'load_unsw_nb15_cleaned',
    'load_hdfs_unlabeled', 'load_unsw_nb15_unlabeled',
    'estimate_contamination',
    
    # Anomaly detection
    'detect_anomalies_unsupervised', 'detect_anomalies_clustering',
    'evaluate_unsupervised_results',
    
    # Advanced evaluation
    'reduce_dimensionality', 'split_embeddings_and_labels',
    'plot_train_test_2d', 'create_comprehensive_evaluation_split',
    'prepare_evaluation_data',
    
    # RAG system
    'RAGEvaluationSystem', 'store_embeddings_in_milvus',
    'rag_explanation_monitored', 'cleanup_milvus_db',
    
    # Supervised evaluation
    'evaluate_classifiers_cv', 'evaluate_extratrees_cv',
    'run_unsw_nb15_traditional_evaluation', 'evaluate_llm_embeddings_cv',
    
    # Main engine
    'EvaluationEngine'
]