"""
Visualization utilities
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

class PlotManager:
    """
    Manages plot creation and saving
    """
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.plots_dir = self.results_dir / "outputs" / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
    def create_2d_visualization(self, embeddings: np.ndarray, labels: np.ndarray, 
                              approach_name: str, dataset_type: str, 
                              plot_type: str = "train_vs_test") -> str:
        """
        Create 2D visualization plot compatible with original system
        
        Args:
            embeddings: Embeddings array
            labels: Labels array  
            approach_name: Name of the approach
            dataset_type: Type of dataset
            plot_type: Type of plot ("train_vs_test", "LLM_train_vs_test", "anomaly_detection")
            
        Returns:
            Path to saved plot file
        """
        safe_name = approach_name.replace(" ", "_").replace("/", "_")
        
        # Handle NaN values
        if np.isnan(embeddings).any():
            logger.warning(f"Found NaN values in embeddings for plotting - indicates embedding generation issue")
            embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reduce to 2D using PCA
        if embeddings.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(embeddings)
            explained_var = pca.explained_variance_ratio_.sum()
            logger.info(f"PCA explained variance: {explained_var:.3f}")
        else:
            embeddings_2d = embeddings
            explained_var = 1.0
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot points with different colors for labels - matching original style
        palette = {0: 'blue', 1: 'red'}
        label_names = {0: 'Normal', 1: 'Anomaly'}
        
        for label_val in np.unique(labels):
            mask = labels == label_val
            sample_count = np.sum(mask)
            label_text = f"{label_names[label_val]} (n={sample_count})"
            
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=palette[label_val], label=label_text, alpha=0.7, s=80)
        
        plt.xlabel('PC1', fontsize=14)
        plt.ylabel('PC2', fontsize=14)
        
        # Add PCA explained variance to title
        title_suffix = f"(PCA: {explained_var:.3f})"
        plt.title(f'{approach_name}: {plot_type.replace("_", " ").title()} 2D PCA {title_suffix}', fontsize=16)
        
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_filename = f"{safe_name}_{plot_type}_2d({dataset_type}).png"
        plot_path = self.plots_dir / plot_filename
        
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved 2D plot: {plot_path}")
        return str(plot_path)
    
    def create_train_test_2d_plot(self, X_train: np.ndarray, y_train: np.ndarray,
                                 X_test: np.ndarray, y_test: np.ndarray,
                                 approach_name: str, dataset_type: str, 
                                 plot_type: str = "LLM_train_vs_test") -> str:
        """
        Create 2D train/test visualization plot exactly matching original pattern
        
        Args:
            X_train, y_train: Training data and labels
            X_test, y_test: Test data and labels
            approach_name: Name of the approach
            dataset_type: Type of dataset
            plot_type: Type of plot
            
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
        
        # Plot training points (filled markers)
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
        
        # Calculate explained variance for title
        explained_var = sum(pca2.explained_variance_ratio_)
        title_suffix = f"(PCA: {explained_var:.3f})"
        
        plt.title(f"{approach_name}: Train vs Test 2D PCA {title_suffix}", fontsize=16)
        plt.xlabel("PC1", fontsize=14)
        plt.ylabel("PC2", fontsize=14)
        plt.legend(fontsize=12, loc='best')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        safe_name = approach_name.replace(" ", "_").replace("/", "_")
        plot_filename = f"{safe_name}_{plot_type}_2d({dataset_type}).png"
        plot_path = self.plots_dir / plot_filename
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved Train/Test 2D scatter to {plot_path}")
        return str(plot_path)
    
    def create_clustering_plots(self, embeddings_2d: np.ndarray, cluster_labels: np.ndarray,
                              approach_name: str, method: str, 
                              anomaly_predictions: Optional[np.ndarray] = None) -> str:
        """
        Create clustering visualization plots (HDBSCAN, DBSCAN)
        
        Args:
            embeddings_2d: 2D embeddings
            cluster_labels: Cluster labels
            approach_name: Name of the approach
            method: Clustering method ("hdbscan", "dbscan")
            anomaly_predictions: Binary anomaly predictions (optional)
            
        Returns:
            Path to saved plot
        """
        safe_name = approach_name.replace(" ", "_").replace("/", "_")
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Cluster visualization
        plt.subplot(1, 2, 1)
        
        # Use different colors for clusters
        unique_clusters = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        
        for i, cluster_id in enumerate(unique_clusters):
            mask = cluster_labels == cluster_id
            if cluster_id == -1:
                # Noise points
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], s=50, alpha=0.7, label=f'Cluster {cluster_id}')
        
        plt.xlabel('Dimension 1')
        plt.ylabel('Dimension 2')
        plt.title(f'{method.upper()} Clustering')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Anomaly detection (if available)
        if anomaly_predictions is not None:
            plt.subplot(1, 2, 2)
            
            normal_mask = anomaly_predictions == 0
            anomaly_mask = anomaly_predictions == 1
            
            plt.scatter(embeddings_2d[normal_mask, 0], embeddings_2d[normal_mask, 1], 
                       c='blue', alpha=0.6, s=50, label='Normal')
            plt.scatter(embeddings_2d[anomaly_mask, 0], embeddings_2d[anomaly_mask, 1], 
                       c='red', alpha=0.8, s=50, label='Anomaly')
            
            plt.xlabel('Dimension 1')
            plt.ylabel('Dimension 2')
            plt.title(f'{method.upper()} Anomaly Detection')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"{method}_clusters_{safe_name}.png"
        plot_path = self.results_dir / plot_filename
        
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved {method} clustering plot: {plot_path}")
        return str(plot_path)

def create_2d_plot(embeddings: np.ndarray, labels: np.ndarray, approach_name: str, 
                  dataset_type: str, save_path: str) -> str:
    """
    Standalone function to create 2D visualization plot
    """
    plot_manager = PlotManager(str(Path(save_path).parent.parent))
    return plot_manager.create_2d_visualization(embeddings, labels, approach_name, dataset_type)

def create_clustering_plots(clustering_results: Dict[str, Any], approach_name: str, 
                          results_dir: str) -> Dict[str, str]:
    """
    Create clustering plots for multiple methods
    
    Args:
        clustering_results: Results from detect_anomalies_clustering
        approach_name: Name of the approach
        results_dir: Results directory
        
    Returns:
        Dictionary mapping method names to plot paths
    """
    plot_paths = {}
    
    for method in ['hdbscan', 'dbscan']:
        if method in clustering_results:
            result = clustering_results[method]
            embeddings_2d = result.get('embeddings_2d')
            cluster_labels = result.get('labels')
            anomaly_predictions = result.get('predictions')
            
            if embeddings_2d is not None and cluster_labels is not None:
                plot_manager = PlotManager(results_dir)
                plot_path = plot_manager.create_clustering_plots(
                    embeddings_2d, cluster_labels, approach_name, method, anomaly_predictions
                )
                plot_paths[method] = plot_path
    
    return plot_paths

def save_anomaly_data(anomaly_predictions: np.ndarray, anomaly_scores: np.ndarray,
                     embeddings: np.ndarray, approach_name: str, results_dir: str) -> str:
    """
    Save anomaly detection data as .npz file (compatible with original)
    
    Args:
        anomaly_predictions: Binary anomaly predictions
        anomaly_scores: Anomaly scores
        embeddings: Original embeddings
        approach_name: Name of the approach
        results_dir: Results directory
        
    Returns:
        Path to saved .npz file
    """
    safe_name = approach_name.replace(" ", "_").replace("/", "_")
    anomaly_file = Path(results_dir) / f"anomalies_{safe_name}_unlabeled.npz"
    
    np.savez_compressed(
        anomaly_file,
        predictions=anomaly_predictions,
        scores=anomaly_scores,
        embeddings=embeddings
    )
    
    logger.info(f"Saved anomaly data: {anomaly_file}")
    return str(anomaly_file)