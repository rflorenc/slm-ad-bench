import os
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

logger = logging.getLogger(__name__)

def load_log_templates(template_path: str) -> Dict[str, str]:
    """
    Load log templates from CSV file and create mapping from EventId to template
    
    Args:
        template_path: Path to HDFS.log_templates.csv file
        
    Returns:
        Dictionary mapping event IDs (e.g., 'E5') to cleaned template descriptions
    """
    try:
        templates_df = pd.read_csv(template_path)
        template_mapping = {}
        
        for _, row in templates_df.iterrows():
            event_id = str(row['EventId']).strip()
            template = str(row['EventTemplate']).strip()
            
            cleaned_template = template.replace('[*]', '').strip()
            cleaned_template = ' '.join(cleaned_template.split())
            
            template_mapping[event_id] = cleaned_template
            
        logger.info(f"Loaded {len(template_mapping)} log templates from {template_path}")
        return template_mapping
        
    except Exception as e:
        logger.warning(f"Could not load log templates from {template_path}: {e}")
        return {}

def enrich_with_templates(event_ids: List[str], template_mapping: Dict[str, str]) -> List[str]:
    """
    Replace event IDs with their semantic templates
    
    Args:
        event_ids: List of event IDs like ['E5', 'E22', 'E11']
        template_mapping: Dictionary mapping event IDs to templates
        
    Returns:
        List of enriched descriptions
    """
    enriched = []
    for event_id in event_ids:
        if event_id in template_mapping:
            enriched.append(template_mapping[event_id])
        else:
            enriched.append(event_id)
    return enriched

def enrich_text_for_rag(text: str, template_mapping: Dict[str, str]) -> str:
    """Enrich text with semantic templates for RAG/reasoning contexts only"""
    if not template_mapping:
        return text
    
    tokens = text.split()
    enriched_tokens = enrich_with_templates(tokens, template_mapping)
    return ' '.join(enriched_tokens)

def load_dataset(csv_path: str, dataset_type: str = "eventtraces", nlines: Optional[int] = None, preprocessing: Optional[Dict] = None) -> Tuple[List[str], np.ndarray]:
    """
    Generic dataset loader dispatcher
    
    Args:
        csv_path: Path to dataset file
        dataset_type: Type of dataset ("eventtraces", "unsw-nb15")  
        nlines: Number of lines to load (None for all)
        preprocessing: Dict of preprocessing options
        
    Returns:
        Tuple of (lines, labels) where lines are text representations
    """
    logger.info(f"Loading {dataset_type} dataset from {csv_path}, nlines={nlines}")
    
    if dataset_type == "eventtraces":
        return load_event_traces(csv_path, nlines, preprocessing)
    elif dataset_type == "unsw-nb15":
        return load_unsw_nb15(csv_path, nlines, preprocessing)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

def load_event_traces(csv_path: str, nlines: Optional[int] = None, preprocessing: Optional[Dict] = None) -> Tuple[List[str], np.ndarray]:
    """
    Load HDFS/EventTraces dataset
    
    Args:
        csv_path: Path to CSV file with Features and Label columns
        nlines: Number of lines to load
        preprocessing: Dict of preprocessing options
        
    Returns:
        Tuple of (lines, labels) where lines are space-separated feature tokens
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if nlines and nlines < len(df):
        logger.info(f"Sampling {nlines} lines from {len(df)} total lines")
        df = df.sample(n=nlines, random_state=42)  # Random sampling like original
    elif nlines:
        logger.info(f"Requested {nlines} lines, but dataset only has {len(df)} lines")
    
    logger.info(f"Loaded EventTraces dataset: {len(df)} samples")
    
    preprocessing = preprocessing or {}
    
    template_mapping = {}
    if preprocessing.get('enable_semantic_enrichment', False):
        template_path = preprocessing.get('template_path', 
            'datasets/intermediate_tasks/task1_systems/loghub/HDFS/HDFS_v1_labelled/preprocessed/HDFS.log_templates.csv')
        if os.path.exists(template_path):
            template_mapping = load_log_templates(template_path)
        else:
            logger.warning(f"Template path not found: {template_path}. Proceeding without semantic enrichment.")
    
    lines = []
    labels = []
    
    for idx, row in df.iterrows():
        features = row['Features']
        label_val = row['Label']
        
        if preprocessing.get('skip_missing_labels', True) and pd.isnull(label_val):
            continue
            
        if pd.notna(features):
            raw_feat = str(features).strip()
            
            if raw_feat.startswith('[') and raw_feat.endswith(']'):
                raw_feat = raw_feat[1:-1]  # Remove brackets

            if preprocessing.get('parse_features', True):
                tokens = [t.strip() for t in raw_feat.split(',') if t.strip() != ""]
                if preprocessing.get('remove_empty_tokens', True):
                    tokens = [t for t in tokens if t != ""]
                
                if tokens:  # Only add if we have tokens
                    lines.append(' '.join(tokens))
                    labels.append(label_val)
            else:
                tokens = str(features).strip('[]').split()
                if tokens:
                    # Keep original event IDs for embeddings/classification  
                    lines.append(' '.join(tokens))
                    labels.append(label_val)
        elif not preprocessing.get('skip_missing_labels', True):
            lines.append("missing")
            labels.append(label_val)
    
    label_mapping = {"Success": 0, "Normal": 0, "Fail": 1, "Anomaly": 1}
    processed_labels = []
    for label in labels:
        if pd.notna(label):
            mapped_label = label_mapping.get(str(label), 1)  # Default to anomaly if unknown
            processed_labels.append(mapped_label)
        else:
            processed_labels.append(0)  # Default to normal for missing labels
    
    return lines, np.array(processed_labels)

def load_unsw_nb15(csv_path: str, nlines: Optional[int] = None, preprocessing: Optional[Dict] = None) -> Tuple[List[str], np.ndarray]:
    """
    Load UNSW-NB15 dataset as text for LLM processing
    
    Args:
        csv_path: Path to UNSW-NB15 CSV file
        nlines: Number of lines to load
        preprocessing: Dict of preprocessing options
        
    Returns:
        Tuple of (lines, labels) where lines are space-separated feature values
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if nlines and nlines < len(df):
        logger.info(f"Sampling {nlines} lines from {len(df)} total lines")
        df = df.sample(n=nlines, random_state=42)  # Random sampling like original
    elif nlines:
        logger.info(f"Requested {nlines} lines, but dataset only has {len(df)} lines")
    
    logger.info(f"Loaded UNSW-NB15 dataset: {len(df)} samples")

    preprocessing = preprocessing or {}

    if preprocessing.get('remove_duplicates', False):
        original_len = len(df)
        df = df.drop_duplicates()
        logger.info(f"Removed {original_len - len(df)} duplicate rows")

    drop_cols = ['id', 'sttl', 'dttl', 'ct_state_ttl']
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    if 'label' in df.columns:
        labels = df['label'].values
        drop_cols.append('label')
    elif 'attack_cat' in df.columns:
        labels = (df['attack_cat'] != 'Normal').astype(int).values
        drop_cols.append('attack_cat')
    else:
        raise ValueError("No label column found in UNSW-NB15 dataset")

    logger.info(f"Dropping columns (research methodology): {drop_cols}")

    feature_df = df.drop(columns=drop_cols, errors='ignore')

    if preprocessing.get('encode_categorical', True):
        categorical_cols = ['proto', 'service', 'state']
        categorical_cols = [col for col in categorical_cols if col in feature_df.columns]
        
        for col in categorical_cols:
            if col in feature_df.columns:
                feature_df[col] = feature_df[col].fillna('unknown')
                le = LabelEncoder()
                feature_df[col] = le.fit_transform(feature_df[col].astype(str))

    if preprocessing.get('handle_missing', True):
        for col in feature_df.columns:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
        feature_df = feature_df.fillna(0)
    
    lines = []
    for _, row in feature_df.iterrows():
        line = ' '.join([str(val) for val in row.values])
        lines.append(line)
    
    return lines, labels

def load_unsw_nb15_cleaned(csv_path: str, label_col: str = 'label', drop_cols: Optional[List[str]] = None, 
                          nlines: Optional[int] = None, multi_class: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UNSW-NB15 dataset with numerical preprocessing for traditional ML
    
    Args:
        csv_path: Path to UNSW-NB15 CSV file
        label_col: Name of label column  
        drop_cols: Additional columns to drop
        nlines: Number of lines to load
        multi_class: Whether to use multi-class labels
        
    Returns:
        Tuple of (X_scaled, y) with normalized features and labels
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if nlines and nlines < len(df):
        logger.info(f"Sampling {nlines} lines from {len(df)} total lines")
        df = df.sample(n=nlines, random_state=42)  # Random sampling like original
    elif nlines:
        logger.info(f"Requested {nlines} lines, but dataset only has {len(df)} lines")
    
    logger.info(f"Loaded UNSW-NB15 cleaned dataset: {len(df)} samples")

    if drop_cols is None:
        drop_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'sttl', 'dttl', 'ct_state_ttl']
    
    if label_col in df.columns:
        if multi_class and label_col == 'attack_cat':
            le = LabelEncoder()
            y = le.fit_transform(df[label_col])
        else:
            y = df[label_col].values
    else:
        raise ValueError(f"Label column '{label_col}' not found")

    drop_cols = drop_cols + [label_col]
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=drop_cols)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Preprocessed features shape: {X_scaled.shape}")
    return X_scaled, y

def load_dataset_unlabeled(data_path: str, dataset_type: str = "hdfs", nlines: Optional[int] = None) -> Tuple[List[str], None, Optional[Dict]]:
    """
    Generic unlabeled dataset loader
    
    Args:
        data_path: Path to dataset file
        dataset_type: Type of dataset ("hdfs", "unsw-nb15")
        nlines: Number of lines to load
        
    Returns:
        Tuple of (lines, None, metadata)
    """
    logger.info(f"Loading unlabeled {dataset_type} dataset from {data_path}")
    
    if dataset_type == "hdfs":
        return load_hdfs_unlabeled(data_path, nlines)
    elif dataset_type == "unsw-nb15":
        return load_unsw_nb15_unlabeled(data_path, nlines)
    else:
        raise ValueError(f"Unknown unlabeled dataset type: {dataset_type}")

def load_hdfs_unlabeled(log_path: str, nlines: Optional[int] = None) -> Tuple[List[str], None, Dict]:
    """
    Load HDFS logs for unlabeled anomaly detection
    
    Args:
        log_path: Path to HDFS log CSV file
        nlines: Number of lines to load
        
    Returns:
        Tuple of (lines, None, metadata)
    """
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"HDFS log file not found: {log_path}")
    
    df = pd.read_csv(log_path)
    if nlines:
        df = df.head(nlines)
    
    logger.info(f"Loaded HDFS logs: {len(df)} samples")
    
    lines = []
    for _, row in df.iterrows():
        log_text_parts = []
        for col in ['Date', 'Time', 'Level', 'Component', 'Content']:
            if col in df.columns and pd.notna(row[col]):
                log_text_parts.append(str(row[col]))
        lines.append(' '.join(log_text_parts))

    metadata = {}
    for col in ['LineId', 'EventId', 'EventTemplate']:
        if col in df.columns:
            metadata[col] = df[col].tolist()
    
    return lines, None, metadata

def load_unsw_nb15_unlabeled(csv_path: str, nlines: Optional[int] = None) -> Tuple[np.ndarray, None, List[str]]:
    """
    Load UNSW-NB15 dataset for unlabeled anomaly detection
    
    Args:
        csv_path: Path to UNSW-NB15 CSV file
        nlines: Number of lines to load
        
    Returns:
        Tuple of (X_scaled, None, lines) with features, no labels, and text representation
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    if nlines and nlines < len(df):
        logger.info(f"Sampling {nlines} lines from {len(df)} total lines")
        df = df.sample(n=nlines, random_state=42)  # Random sampling like original
    elif nlines:
        logger.info(f"Requested {nlines} lines, but dataset only has {len(df)} lines")
    
    logger.info(f"Loaded UNSW-NB15 unlabeled dataset: {len(df)} samples")
    
    # Handle different column counts (47 vs 49 columns) - unsw_nb15_unlabeled specific!
    if len(df.columns) == 49:
        label_cols = ['label', 'attack_cat']
        df = df.drop(columns=[col for col in label_cols if col in df.columns])
    
    drop_cols = ['id', 'srcip', 'sport', 'dstip', 'dsport', 'stime', 'ltime', 'sttl', 'dttl', 'ct_state_ttl']
    drop_cols = [col for col in drop_cols if col in df.columns]
    X = df.drop(columns=drop_cols)
    
    lines = []
    for _, row in X.iterrows():
        line = ' '.join([str(val) for val in row.values])
        lines.append(line)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    logger.info(f"Preprocessed unlabeled features shape: {X_scaled.shape}")
    return X_scaled, None, lines

def estimate_contamination(data: np.ndarray, method: str = 'statistical', k: int = 10) -> float:
    """
    Estimate contamination rate for unsupervised anomaly detection
    
    Args:
        data: Input data array
        method: Estimation method ('statistical', 'elbow', 'density')
        k: Number of neighbors for density method
        
    Returns:
        Estimated contamination rate between 0.001 and 0.5
    """
    if method == 'statistical':
        median = np.median(data, axis=0)
        mad = np.median(np.abs(data - median), axis=0)
        modified_z_scores = 0.6745 * (data - median) / mad
        outliers = np.any(np.abs(modified_z_scores) > 3.5, axis=1)
        contamination = np.mean(outliers)
    
    elif method == 'elbow':
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(data)
        distances, _ = nbrs.kneighbors(data)
        k_distances = distances[:, k].copy()
        k_distances.sort()
        
        diffs = np.diff(k_distances)
        elbow_idx = np.argmax(diffs)
        contamination = 1.0 - (elbow_idx / len(k_distances))
    
    elif method == 'density':
        lof = LocalOutlierFactor(n_neighbors=k, contamination='auto')
        outlier_scores = lof.fit_predict(data)
        contamination = np.mean(outlier_scores == -1)
    
    else:
        raise ValueError(f"Unknown contamination estimation method: {method}")

    contamination = max(0.001, min(0.5, contamination))
    logger.info(f"Estimated contamination using {method}: {contamination:.3f}")
    
    return contamination