"""
Unsupervised anomaly detection functions
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, Optional, Tuple, List
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)

def detect_anomalies_unsupervised(embeddings: np.ndarray, contamination: float = 0.1, 
                                method: str = 'ensemble') -> Dict[str, Any]:
    """
    Detect anomalies using unsupervised methods
    
    Args:
        embeddings: Input embeddings array of shape (n_samples, n_features)
        contamination: Expected proportion of anomalies (0.0 to 0.5)
        method: Detection method ('ensemble', 'IsolationForest', 'LocalOutlierFactor', 
               'OneClassSVM', 'EllipticEnvelope')
               
    Returns:
        Dictionary with predictions, scores, and method-specific results
    """
    logger.info(f"Running unsupervised anomaly detection with method={method}, contamination={contamination}")
    
    if embeddings.shape[0] == 0:
        raise ValueError("Empty embeddings array")
    
    # Available methods
    methods = {
        'IsolationForest': IsolationForest(contamination=contamination, random_state=42, n_jobs=-1),
        'LocalOutlierFactor': LocalOutlierFactor(contamination=contamination, n_jobs=-1),
        'OneClassSVM': OneClassSVM(nu=contamination),
        'EllipticEnvelope': EllipticEnvelope(contamination=contamination, random_state=42)
    }
    
    results = {
        'individual_predictions': {},
        'scores': {},
        'method_info': {}
    }
    
    if method == 'ensemble':
        # Run all methods and use majority voting
        logger.info("Running ensemble anomaly detection")
        all_predictions = []
        all_scores = []
        
        for method_name, detector in methods.items():
            try:
                logger.debug(f"Running {method_name}")
                
                if method_name == 'LocalOutlierFactor':
                    # LOF doesn't have decision_function, use negative_outlier_factor_
                    predictions = detector.fit_predict(embeddings)
                    scores = detector.negative_outlier_factor_
                else:
                    detector.fit(embeddings)
                    predictions = detector.predict(embeddings)
                    if hasattr(detector, 'decision_function'):
                        scores = detector.decision_function(embeddings)
                    elif hasattr(detector, 'score_samples'):
                        scores = detector.score_samples(embeddings)
                    else:
                        scores = predictions.astype(float)
                
                # Convert to binary (1 for anomaly, 0 for normal)
                binary_predictions = (predictions == -1).astype(int)
                
                all_predictions.append(binary_predictions)
                all_scores.append(scores)
                
                results['individual_predictions'][method_name] = binary_predictions
                results['scores'][method_name] = scores
                results['method_info'][method_name] = {
                    'anomalies_detected': np.sum(binary_predictions),
                    'anomaly_ratio': np.mean(binary_predictions)
                }
                
            except Exception as e:
                logger.warning(f"Failed to run {method_name}: {e}")
                continue
        
        if all_predictions:
            # ensemble prediction using majority voting
            predictions_array = np.array(all_predictions)
            ensemble_predictions = (np.mean(predictions_array, axis=0) >= 0.5).astype(int)
            
            # ensemble scores as average of normalized scores
            normalized_scores = []
            for scores in all_scores:
                # normalize scores
                scores_norm = (scores - np.min(scores)) / (np.max(scores) - np.min(scores) + 1e-8)
                normalized_scores.append(scores_norm)
            ensemble_scores = np.mean(normalized_scores, axis=0)
            
            results['ensemble'] = ensemble_predictions
            results['ensemble_scores'] = ensemble_scores
            results['method_info']['ensemble'] = {
                'anomalies_detected': np.sum(ensemble_predictions),
                'anomaly_ratio': np.mean(ensemble_predictions),
                'methods_used': len(all_predictions)
            }
        else:
            raise RuntimeError("All anomaly detection methods failed")
            
    else:
        # single method run
        if method not in methods:
            raise ValueError(f"Unknown method: {method}. Available: {list(methods.keys())}")
        
        detector = methods[method]
        logger.info(f"Running {method} anomaly detection")
        
        if method == 'LocalOutlierFactor':
            predictions = detector.fit_predict(embeddings)
            scores = detector.negative_outlier_factor_
        else:
            detector.fit(embeddings)
            predictions = detector.predict(embeddings)
            if hasattr(detector, 'decision_function'):
                scores = detector.decision_function(embeddings)
            elif hasattr(detector, 'score_samples'):
                scores = detector.score_samples(embeddings)
            else:
                scores = predictions.astype(float)

        binary_predictions = (predictions == -1).astype(int)
        
        results[method] = binary_predictions
        results['scores'][method] = scores
        results['method_info'][method] = {
            'anomalies_detected': np.sum(binary_predictions),
            'anomaly_ratio': np.mean(binary_predictions)
        }
    
    logger.info(f"Anomaly detection completed. Results: {results.get('method_info', {})}")
    return results

def detect_anomalies_clustering(embeddings: np.ndarray, contamination: float = 0.1, 
                               plot_root: Optional[str] = None, approach_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Detect anomalies using clustering methods (HDBSCAN and DBSCAN)
    
    Args:
        embeddings: Input embeddings array
        contamination: Expected proportion of anomalies
        plot_root: Directory to save plots (optional)
        approach_name: Name for plot titles (optional)
        
    Returns:
        Dictionary with clustering results and anomaly predictions
    """
    logger.info(f"Running clustering-based anomaly detection with contamination={contamination}")
    
    if embeddings.shape[0] == 0:
        raise ValueError("Empty embeddings array")
    
    results = {}
    
    # PCA to 2D for clustering visual
    if embeddings.shape[1] > 2:
        logger.info("Reducing dimensionality to 2D using PCA")
        pca = PCA(n_components=2, random_state=42)
        embeddings_2d = pca.fit_transform(embeddings)
        logger.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    else:
        embeddings_2d = embeddings.copy()
    
    # HDBSCAN clustering
    try:
        import hdbscan
        
        min_cluster_size = max(5, int(len(embeddings) * 0.01))
        logger.info(f"Running HDBSCAN with min_cluster_size={min_cluster_size}")
        
        hdb = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=1,
            cluster_selection_epsilon=0.0,
            alpha=1.0
        )
        
        cluster_labels = hdb.fit_predict(embeddings_2d)
        
        # Anomalies are points labeled as -1 / noise
        hdb_predictions = (cluster_labels == -1).astype(int)
        
        # Use outlier scores if available
        if hasattr(hdb, 'outlier_scores_'):
            outlier_scores = hdb.outlier_scores_
        else:
            outlier_scores = hdb_predictions.astype(float)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        results['hdbscan'] = {
            'predictions': hdb_predictions,
            'scores': outlier_scores,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'embeddings_2d': embeddings_2d,
            'anomalies_detected': np.sum(hdb_predictions),
            'anomaly_ratio': np.mean(hdb_predictions)
        }
        
        logger.info(f"HDBSCAN: {n_clusters} clusters, {np.sum(hdb_predictions)} anomalies")
        
    except ImportError:
        logger.warning("HDBSCAN not available, skipping")
    except Exception as e:
        logger.warning(f"HDBSCAN failed: {e}")
    
    # DBSCAN clustering
    try:
        # Estimate eps using k-distance graph
        from sklearn.neighbors import NearestNeighbors
        
        k = 5
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings_2d)
        distances, _ = nbrs.kneighbors(embeddings_2d)
        k_distances = distances[:, k-1]
        k_distances.sort()
        
        # Use 95th percentile as eps
        eps = np.percentile(k_distances, 95)
        min_samples = max(3, int(len(embeddings) * 0.005))
        
        logger.info(f"Running DBSCAN with eps={eps:.4f}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings_2d)
        
        dbscan_predictions = (cluster_labels == -1).astype(int)
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        results['dbscan'] = {
            'predictions': dbscan_predictions,
            'labels': cluster_labels,
            'n_clusters': n_clusters,
            'embeddings_2d': embeddings_2d,
            'eps': eps,
            'min_samples': min_samples,
            'anomalies_detected': np.sum(dbscan_predictions),
            'anomaly_ratio': np.mean(dbscan_predictions)
        }
        
        logger.info(f"DBSCAN: {n_clusters} clusters, {np.sum(dbscan_predictions)} anomalies")
        
    except Exception as e:
        logger.warning(f"DBSCAN failed: {e}")
    
    if not results:
        raise RuntimeError("All clustering methods failed")
    
    return results

def evaluate_unsupervised_results(embeddings: np.ndarray, predictions: np.ndarray, 
                                clustering_labels: Optional[np.ndarray] = None,
                                anomaly_scores: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Evaluate unsupervised anomaly detection results
    
    Args:
        embeddings: Original embeddings
        predictions: Binary anomaly predictions (1 for anomaly, 0 for normal)
        clustering_labels: Cluster labels (optional)
        anomaly_scores: Anomaly scores (optional)
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating unsupervised anomaly detection results")
    
    metrics = {}

    n_total = len(predictions)
    n_anomalies = np.sum(predictions)
    n_normal = n_total - n_anomalies
    anomaly_ratio = n_anomalies / n_total if n_total > 0 else 0
    
    metrics.update({
        'num_samples': n_total,
        'num_anomalies': int(n_anomalies),
        'num_normal': int(n_normal),
        'anomaly_ratio': anomaly_ratio
    })

    if clustering_labels is not None and len(set(clustering_labels)) > 1:
        try:
            # remove noise points for clustering metrics
            valid_mask = clustering_labels != -1
            if np.sum(valid_mask) > 1:
                valid_embeddings = embeddings[valid_mask]
                valid_labels = clustering_labels[valid_mask]
                
                if len(set(valid_labels)) > 1:
                    silhouette = silhouette_score(valid_embeddings, valid_labels)
                    davies_bouldin = davies_bouldin_score(valid_embeddings, valid_labels)
                    
                    metrics.update({
                        'silhouette_score': silhouette,
                        'davies_bouldin_score': davies_bouldin,
                        'n_clusters': len(set(valid_labels))
                    })
                    
                    logger.info(f"Clustering metrics - Silhouette: {silhouette:.3f}, "
                              f"Davies-Bouldin: {davies_bouldin:.3f}")
        except Exception as e:
            logger.warning(f"Failed to compute clustering metrics: {e}")
    
    # anomaly score stats
    if anomaly_scores is not None:
        try:
            score_stats = {
                'score_mean': np.mean(anomaly_scores),
                'score_std': np.std(anomaly_scores),
                'score_min': np.min(anomaly_scores),
                'score_max': np.max(anomaly_scores),
                'score_median': np.median(anomaly_scores),
                'score_25th': np.percentile(anomaly_scores, 25),
                'score_75th': np.percentile(anomaly_scores, 75),
                'score_95th': np.percentile(anomaly_scores, 95),
                'score_99th': np.percentile(anomaly_scores, 99)
            }
            metrics.update(score_stats)
            
            # score stats by prediction
            if n_anomalies > 0 and n_normal > 0:
                anomaly_scores_pos = anomaly_scores[predictions == 1]
                anomaly_scores_neg = anomaly_scores[predictions == 0]
                
                metrics.update({
                    'score_anomaly_mean': np.mean(anomaly_scores_pos),
                    'score_normal_mean': np.mean(anomaly_scores_neg),
                    'score_separation': np.mean(anomaly_scores_pos) - np.mean(anomaly_scores_neg)
                })
                
        except Exception as e:
            logger.warning(f"Failed to compute score statistics: {e}")
    
    # embedding stats
    try:
        embedding_stats = {
            'embedding_dim': embeddings.shape[1],
            'embedding_mean_norm': np.mean(np.linalg.norm(embeddings, axis=1)),
            'embedding_std_norm': np.std(np.linalg.norm(embeddings, axis=1))
        }
        metrics.update(embedding_stats)
    except Exception as e:
        logger.warning(f"Failed to compute embedding statistics: {e}")
    
    logger.info(f"Unsupervised evaluation metrics computed: {len(metrics)} metrics")
    return metrics

def reduce_dimensionality(embeddings: np.ndarray, target_dim: int = 768, 
                         method: str = 'pca') -> Tuple[np.ndarray, Any]:
    """
    Reduce dimensionality of embeddings
    
    Args:
        embeddings: Input embeddings array
        target_dim: Target number of dimensions
        method: Dimensionality reduction method ('pca')
        
    Returns:
        Tuple of (reduced_embeddings, reducer_object)
    """
    if embeddings.shape[1] <= target_dim:
        logger.info(f"Embeddings already have {embeddings.shape[1]} dimensions, no reduction needed")
        return embeddings, None
    
    logger.info(f"Reducing embeddings from {embeddings.shape[1]} to {target_dim} dimensions using {method}")
    
    if method == 'pca':
        reducer = PCA(n_components=target_dim, random_state=42)
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        explained_variance = np.sum(reducer.explained_variance_ratio_)
        logger.info(f"PCA explained variance ratio: {explained_variance:.3f}")
        
        return reduced_embeddings, reducer
    else:
        raise ValueError(f"Unknown dimensionality reduction method: {method}")