"""
Includes PCA reduction, train/test splitting, and comprehensive monitoring
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger(__name__)

def reduce_dimensionality(embeddings: np.ndarray, target_dim: int = 768, method: str = 'pca') -> Tuple[np.ndarray, Any]:
    """
    Reduce embedding dimensionality using specified method
    
    Args:
        embeddings: Input embeddings of shape (n_samples, n_features)
        target_dim: Target dimensionality 
        method: Reduction method ('pca')
    
    Returns:
        Tuple of (reduced_embeddings, reducer_object)
    """
    d0 = embeddings.shape[1]
    
    if d0 <= target_dim:
        logger.info(f"Embeddings already have dimension {d0} <= {target_dim}, no reduction needed")
        return embeddings, None
    
    logger.info(f"Reducing from {d0} to {target_dim} via {method}...")
    
    if method == 'pca':
        reducer = PCA(n_components=target_dim, random_state=42)
        emb_reduced = reducer.fit_transform(embeddings)
        explained_variance = sum(reducer.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {explained_variance:.4f}")
        return emb_reduced, reducer
    else:
        raise ValueError(f"Unknown reduction method: {method}")

def split_embeddings_and_labels(embeddings: np.ndarray, labels: np.ndarray, 
                               test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split embeddings and labels into train/test sets
    
    Args:
        embeddings: Input embeddings
        labels: Corresponding labels
        test_size: Fraction for test set
        random_state: Random seed
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data: {len(embeddings)} samples, test_size={test_size}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, 
        test_size=test_size, 
        random_state=random_state,
        stratify=labels
    )
    
    logger.info(f"Train set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test

def plot_train_test_2d(X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray, 
                      approach_name: str, dataset_type: str, 
                      output_dir: str) -> str:
    """
    Create 2D PCA plot with train/test visualization
    
    Args:
        X_train, y_train: Training data and labels
        X_test, y_test: Test data and labels  
        approach_name: Name of the approach
        dataset_type: Type of dataset
        output_dir: Output directory for plots
    
    Returns:
        Path to saved plot file
    """
    logger.info("Creating 2D PCA train/test visualization")
    
    # 2D PCA on the TRAIN set, then overlay TEST points with open markers
    pca2 = PCA(n_components=2, random_state=42)
    emb2_train = pca2.fit_transform(X_train)
    emb2_test = pca2.transform(X_test)
    
    plt.figure(figsize=(10, 8))
    palette = {0: 'blue', 1: 'red'}
    label_names = {0: 'Normal', 1: 'Anomaly'}
    
    # Plot trai points (filled markers)
    for label_val in np.unique(y_train):
        idx = y_train == label_val
        plt.scatter(
            emb2_train[idx, 0], emb2_train[idx, 1],
            c=palette[label_val],
            label=f"Train {label_names[label_val]} (n={sum(idx)})",
            s=80, alpha=0.7
        )
    
    # Plot test points (open markers)
    for label_val in np.unique(y_test):
        idx = y_test == label_val
        plt.scatter(
            emb2_test[idx, 0], emb2_test[idx, 1],
            facecolors='none',
            edgecolors=palette[label_val],
            label=f"Test  {label_names[label_val]} (n={sum(idx)})",
            s=120, linewidths=1.5
        )

    explained_var = sum(pca2.explained_variance_ratio_)
    title_suffix = f"(PCA: {explained_var:.3f})"
    
    plt.title(f"{approach_name}: Train vs Test 2D PCA {title_suffix}", fontsize=16)
    plt.xlabel("PC1", fontsize=14)
    plt.ylabel("PC2", fontsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.grid(True, alpha=0.3)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    safe_name = approach_name.replace(" ", "_").replace("/", "_")
    fname = f"{safe_name}_train_vs_test_2d({dataset_type}).png"
    fullpath = output_dir / fname
    
    plt.tight_layout()
    plt.savefig(fullpath, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved Train/Test 2D scatter to {fullpath}")
    return str(fullpath)

def create_comprehensive_evaluation_split(embeddings: np.ndarray, labels: np.ndarray,
                                        approach_name: str, dataset_type: str,
                                        output_dir: str, target_dim: int = 768) -> Dict[str, Any]:
    """
    Evaluation with dimensionality reduction and train/test split
    
    Args:
        embeddings: Input embeddings
        labels: Corresponding labels
        approach_name: Name of the approach
        dataset_type: Dataset type
        output_dir: Output directory
        target_dim: Target dimension for PCA reduction
    
    Returns:
        Dictionary with evaluation results and metadata
    """
    results = {}
    
    # Step 1: Reduce dimensionality if needed
    logger.info(f"Starting comprehensive evaluation for {approach_name}")
    original_dim = embeddings.shape[1]
    
    if original_dim > target_dim:
        X_reduced, pca_reducer = reduce_dimensionality(embeddings, target_dim)
        pca_explained_variance = sum(pca_reducer.explained_variance_ratio_)
        results['pca_applied'] = True
        results['pca_explained_variance'] = float(pca_explained_variance)
        results['original_dim'] = original_dim
        results['reduced_dim'] = target_dim
    else:
        X_reduced = embeddings
        pca_explained_variance = 1.0
        results['pca_applied'] = False
        results['pca_explained_variance'] = 1.0
        results['original_dim'] = original_dim
        results['reduced_dim'] = original_dim
    
    # Step 2: Train/test split
    X_train, X_test, y_train, y_test = split_embeddings_and_labels(X_reduced, labels)
    
    # Step 3: Create 2D visualization
    plot_path = plot_train_test_2d(
        X_train, y_train, X_test, y_test,
        approach_name, dataset_type, 
        os.path.join(output_dir, "plots")
    )
    
    results.update({
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_anomaly_ratio': float(np.mean(y_train)),
        'test_anomaly_ratio': float(np.mean(y_test)),
        'plot_path': plot_path,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    })
    
    logger.info(f"Comprehensive evaluation setup complete for {approach_name}")
    return results

def prepare_evaluation_data(embeddings: np.ndarray, labels: np.ndarray, 
                          approach_name: str, dataset_type: str,
                          results_dir: str) -> Dict[str, Any]:
    """
    Prepare data for evaluation following original pattern exactly
    
    This function matches the original workflow:
    1. PCA reduction from high-dim to 768 if needed
    2. Create train/test split
    3. Generate 2D visualization
    4. Return data ready for classifier evaluation
    """
    
    # Log initial data info
    logger.info(f"Preparing evaluation data for {approach_name}")
    logger.info(f"Initial embeddings shape: {embeddings.shape}")
    logger.info(f"Label distribution: Normal={np.sum(labels==0)}, Anomaly={np.sum(labels==1)}")
    
    # PCA reduction
    if embeddings.shape[1] > 768:
        logger.info("Reducing from {} to 768 via pca...".format(embeddings.shape[1]))
        pca = PCA(n_components=768, random_state=42)
        X_reduced = pca.fit_transform(embeddings)
        pca_explained_var = sum(pca.explained_variance_ratio_)
        logger.info(f"PCA explained variance: {pca_explained_var:.4f}")
    else:
        X_reduced = embeddings
        pca_explained_var = 1.0000
        logger.info(f"No PCA reduction needed (dim={embeddings.shape[1]})")
    
    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, labels, test_size=0.2, random_state=42, stratify=labels
    )

    plots_dir = Path(results_dir) / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = plot_train_test_2d(
        X_train, y_train, X_test, y_test,
        approach_name, dataset_type, str(plots_dir)
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'pca_explained_variance': pca_explained_var,
        'plot_path': plot_path,
        'original_embeddings_shape': embeddings.shape,
        'reduced_embeddings_shape': X_reduced.shape
    }