"""
Supervised evaluation functions
"""

import numpy as np
import pandas as pd
import logging
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate

# Suppress sklearn warnings about precision/recall for imbalanced classes
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA

from .data_loading import load_unsw_nb15_cleaned

logger = logging.getLogger(__name__)

def evaluate_classifiers_cv(embeddings: np.ndarray, labels: np.ndarray, 
                           folds: int = 5, results_dir: str = "outputs") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate multiple classifiers using cross-validation on embeddings
    
    Args:
        embeddings: Feature embeddings array
        labels: Ground truth labels
        folds: Number of CV folds
        results_dir: Directory to save results (not used in this implementation)
        
    Returns:
        Tuple of (detailed_results, summary_results)
    """
    logger.info(f"Evaluating classifiers with {folds}-fold CV on {len(embeddings)} samples")
    
    # Define classifiers with class weight balancing
    classifiers = {
        'LogReg': LogisticRegression(
            random_state=42, 
            max_iter=2000, 
            C=1.0,
            class_weight='balanced',  # Handle class imbalance
            solver='lbfgs'  # Good for small datasets
        ),
        'DecisionTree': DecisionTreeClassifier(
            random_state=42, 
            max_depth=10,
            class_weight='balanced'  # Handle class imbalance
        ),
        'RandomForest': RandomForestClassifier(
            random_state=42, 
            n_estimators=100,
            class_weight='balanced'  # Handle class imbalance
        )
    }
    
    # Metrics to compute
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    results = {}
    summary = {}
    
    for clf_name, clf in classifiers.items():
        logger.info(f"  --> Evaluating {clf_name} with {folds}-fold CV")
        
        try:
            start_time = time.time()
            
            # Perform cross-validation
            cv_results = cross_validate(
                clf, embeddings, labels,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42),
                scoring=scoring,
                return_train_score=False,
                n_jobs=1
            )
            
            eval_time = time.time() - start_time
            
            # Calculate metrics
            metrics = {}
            for metric in scoring:
                test_scores = cv_results[f'test_{metric}']
                metrics[f'{metric}_mean'] = np.mean(test_scores)
                metrics[f'{metric}_std'] = np.std(test_scores)
                metrics[f'{metric}_scores'] = test_scores.tolist()
            
            # Add timing information
            metrics['eval_time'] = eval_time
            metrics['eval_time_per_sample'] = eval_time / len(embeddings)
            
            # Get predictions for additional metrics
            y_pred = cross_val_predict(
                clf, embeddings, labels,
                cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            )
            
            # Calculate PR AUC
            try:
                pr_auc = average_precision_score(labels, y_pred)
                metrics['pr_auc'] = pr_auc
            except Exception as e:
                logger.warning(f"Failed to compute PR AUC for {clf_name}: {e}")
                metrics['pr_auc'] = 0.0
            
            results[clf_name] = metrics
            
            # Summary for this classifier
            summary[clf_name] = {
                'accuracy': metrics['accuracy_mean'],
                'f1': metrics['f1_mean'],
                'roc_auc': metrics['roc_auc_mean'],
                'eval_time': eval_time
            }
            
            logger.info(f"{clf_name} - Accuracy: {metrics['accuracy_mean']:.3f} ± {metrics['accuracy_std']:.3f}, "
                       f"F1: {metrics['f1_mean']:.3f} ± {metrics['f1_std']:.3f}")
            
        except Exception as e:
            logger.error(f"Failed to evaluate {clf_name}: {e}")
            results[clf_name] = {'error': str(e)}
            summary[clf_name] = {'error': str(e)}
    
    return results, summary

def evaluate_extratrees_cv(X: np.ndarray, y: np.ndarray, n_splits: int = 5, 
                          multi_class: bool = False) -> Dict[str, Any]:
    """
    Evaluate ExtraTreesClassifier using cross-validation with custom metrics
    
    Args:
        X: Feature matrix
        y: Labels
        n_splits: Number of CV splits
        multi_class: Whether to use multi-class classification
        
    Returns:
        Dictionary with averaged metrics
    """
    logger.info(f"Evaluating ExtraTrees with {n_splits}-fold CV, multi_class={multi_class}")
    
    # Compute class weights for imbalanced data
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Configure classifier
    clf = ExtraTreesClassifier(
        n_estimators=50,
        random_state=42,
        class_weight=class_weight_dict,
        n_jobs=-1
    )
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_metrics = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit
        start_time = time.time()
        clf.fit(X_train, y_train)
        
        # Predict
        y_pred = clf.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred, average='binary' if not multi_class else 'weighted'),
            'precision': precision_score(y_test, y_pred, average='binary' if not multi_class else 'weighted'),
            'recall': recall_score(y_test, y_pred, average='binary' if not multi_class else 'weighted'),
            'prediction_time_per_sample_us': (prediction_time / len(y_test)) * 1e6  # microseconds
        }
        
        # ROC AUC
        try:
            if multi_class:
                y_proba = clf.predict_proba(X_test)
                if y_proba.shape[1] > 2:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba, multi_class='ovr')
                else:
                    metrics['roc_auc'] = roc_auc_score(y_test, y_proba[:, 1])
            else:
                y_proba = clf.predict_proba(X_test)[:, 1]
                metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
        except Exception as e:
            logger.warning(f"Failed to compute ROC AUC for fold {fold_idx}: {e}")
            metrics['roc_auc'] = 0.0
        
        # Custom metrics: 
        # DR (Detection Rate) and FAR (False Alarm Rate)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel() if not multi_class else (0, 0, 0, 0)
        if not multi_class:
            metrics['dr'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # True Positive Rate
            metrics['far'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
        
        fold_metrics.append(metrics)
        
        logger.debug(f"Fold {fold_idx + 1}: Accuracy={metrics['accuracy']:.3f}, "
                    f"F1={metrics['f1']:.3f}, AUC={metrics['roc_auc']:.3f}")
    
    # Average metrics across folds (convert numpy types to Python types for JSON)
    final_metrics = {}
    for metric_name in fold_metrics[0].keys():
        values = [fold[metric_name] for fold in fold_metrics]
        final_metrics[metric_name] = float(np.mean(values))
        final_metrics[f'{metric_name}_std'] = float(np.std(values))
    
    # update metadata metrics
    final_metrics.update({
        'n_splits': n_splits,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'n_classes': len(classes),
        'class_distribution': np.bincount(y).tolist(),
        'classifier': 'ExtraTreesClassifier',
        'n_estimators': 50
    })
    
    logger.info(f"ExtraTrees CV results - Accuracy: {final_metrics['accuracy']:.3f} ± {final_metrics['accuracy_std']:.3f}, "
               f"F1: {final_metrics['f1']:.3f} ± {final_metrics['f1_std']:.3f}, "
               f"AUC: {final_metrics['roc_auc']:.3f} ± {final_metrics['roc_auc_std']:.3f}")
    
    return final_metrics

def run_unsw_nb15_traditional_evaluation(train_path: str, test_path: str, 
                                       nlines: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Run traditional ML evaluation on UNSW-NB15 dataset
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        nlines: Number of lines to load from each file
        
    Returns:
        List of result dictionaries with evaluation metrics
    """
    logger.info(f"Running UNSW-NB15 traditional evaluation, nlines={nlines}")
    
    try:
        # Load training and test data
        X_train, y_train = load_unsw_nb15_cleaned(train_path, nlines=nlines)
        X_test, y_test = load_unsw_nb15_cleaned(test_path, nlines=nlines)
        
        logger.info(f"Loaded train: {X_train.shape}, test: {X_test.shape}")
        
        # Combine for cross-validation
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.hstack([y_train, y_test])
        
        logger.info(f"Combined dataset shape: {X_combined.shape}")
        
        # Run binary classification
        binary_results = evaluate_extratrees_cv(X_combined, y_combined, n_splits=5, multi_class=False)
        binary_results['classification_type'] = 'binary'
        
        results = [binary_results]
        
        # Try multi-class if attack_cat column is available
        # (check by trying to load)
        try:
            _, y_train_mc = load_unsw_nb15_cleaned(train_path, label_col='attack_cat', nlines=nlines, multi_class=True)
            _, y_test_mc = load_unsw_nb15_cleaned(test_path, label_col='attack_cat', nlines=nlines, multi_class=True)
            
            y_combined_mc = np.hstack([y_train_mc, y_test_mc])
            
            if len(np.unique(y_combined_mc)) > 2:
                logger.info("Running multi-class evaluation")
                mc_results = evaluate_extratrees_cv(X_combined, y_combined_mc, n_splits=5, multi_class=True)
                mc_results['classification_type'] = 'multiclass'
                results.append(mc_results)
                
        except Exception as e:
            logger.info(f"Multi-class evaluation not available: {e}")
        
        logger.info(f"Traditional evaluation completed with {len(results)} result(s)")
        return results
        
    except Exception as e:
        logger.error(f"Traditional evaluation failed: {e}")
        return []

def evaluate_llm_embeddings_cv(embeddings: np.ndarray, labels: np.ndarray, 
                              approach_name: str, dataset_type: str = "unsw-nb15") -> List[Dict[str, Any]]:
    """
    Evaluate LLM embeddings using cross-validation with multiple classifiers
    
    Args:
        embeddings: LLM-generated embeddings
        labels: Ground truth labels
        approach_name: Name of the approach
        dataset_type: Type of dataset
        
    Returns:
        List of result dictionaries
    """
    logger.info(f"Evaluating LLM embeddings for {approach_name} on {dataset_type}")
    
    results = []
    
    try:
        # NaN handling should be done at embedding generation level with random fallbacks
        if np.isnan(embeddings).any():
            logger.warning(f"Found {np.isnan(embeddings).sum()} NaN values in embeddings - this indicates an issue with embedding generation")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply PCA if embeddings are high-dimensional
        
        
        embeddings_processed = embeddings
        pca_applied = False
        
        if embeddings.shape[1] > 768:
            logger.info(f"Applying PCA to reduce from {embeddings.shape[1]} to 768 dimensions")
            pca = PCA(n_components=768, random_state=42)
            embeddings_processed = pca.fit_transform(embeddings)
            pca_applied = True
            explained_variance = np.sum(pca.explained_variance_ratio_)
            logger.info(f"PCA explained variance: {explained_variance:.3f}")
        
        # Evaluate with multiple classifiers
        detailed_results, summary = evaluate_classifiers_cv(embeddings_processed, labels, folds=5)
        
        # result formatign
        for clf_name, clf_results in detailed_results.items():
            if 'error' not in clf_results:
                classifier_mapping = {
                    'LogisticRegression': 'LogReg',
                    'DecisionTree': 'DecisionTree',
                    'RandomForest': 'RandomForest'
                }
                
                result = {
                    'approach': f"{approach_name}_llm",
                    'classifier': classifier_mapping.get(clf_name, clf_name),
                    'acc': clf_results['accuracy_mean'],
                    'f1': clf_results['f1_mean'],
                    'roc_auc': clf_results['roc_auc_mean'],
                    'eval_type': 'research_compliant_llm_embeddings',
                    'methodology': '5_fold_cv_minmax_plus_pca' if pca_applied else '5_fold_cv_minmax_no_pca',
                    'dataset_type': dataset_type
                }
                
                if pca_applied:
                    result['pca_explained_variance'] = explained_variance
                
                results.append(result)
                
                logger.info(f"{clf_name}: Acc={result['acc']:.3f}, F1={result['f1']:.3f}, AUC={result['roc_auc']:.3f}")
        
        if not results:
            logger.warning("No successful classifier evaluations")
            
    except Exception as e:
        logger.error(f"LLM embeddings evaluation failed: {e}")
    
    return results