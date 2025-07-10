"""
File and directory utilities
"""

import os
import json
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from .json_utils import safe_json_dump, ensure_json_serializable

sanitize_for_json = ensure_json_serializable

logger = logging.getLogger(__name__)

def init_run_dir(base_results: str = "output_results", run_type: Optional[str] = None) -> str:
    """Create and return a unique results directory for this run"""
    
    base_dir = Path(base_results)
    base_dir.mkdir(parents=True, exist_ok=True)
    
    date_folder = datetime.now().strftime("%Y-%m-%d")
    date_dir = base_dir / date_folder
    date_dir.mkdir(exist_ok=True)
    
    existing_runs = list(date_dir.glob("run_*"))
    run_idx = len(existing_runs) + 1
    
    timestamp = datetime.now().strftime("%H%M%S")
    
    if run_type:
        run_dir = date_dir / f"run_{run_idx}_{timestamp}_{run_type}"
    else:
        run_dir = date_dir / f"run_{run_idx}_{timestamp}"
    
    run_dir.mkdir(exist_ok=True)
    
    logger.info(f"Created run directory: {run_dir}")
    return str(run_dir)

def write_run_info(results_dir: str, dataset_info: str, nlines: int, dataset_type: str = "eventtraces") -> None:
    """Write run information to run_info.json"""
    
    info = {
        "dataset": dataset_info,
        "nlines": nlines,
        "dataset_type": dataset_type,
        "timestamp": datetime.now().isoformat(),
        "run_type": "proper_train_test_split" if "train:" in str(dataset_info) else "random_split"
    }
    
    run_info_path = Path(results_dir) / "run_info.json"
    with open(run_info_path, "w") as f:
        sanitized_info = sanitize_for_json(info)
        safe_json_dump(sanitized_info, f)
    
    logger.info(f"Wrote run info to {run_info_path}")

def combine_results(results_dir: str, pattern: str = "results_*.json", out_name: str = "combined_results.csv"):
    """
    Combine all JSON result files into a single CSV
    
    Args:
        results_dir: Directory containing result files
        pattern: Glob pattern to match result files
        out_name: Output CSV filename
    
    Returns:
        DataFrame with combined results or None if no results found
    """
    try:
        import pandas as pd
    except ImportError:
        logger.error("pandas is required for combine_results")
        return None
    
    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        logger.warning(f"No files found for pattern {pattern} in {results_dir}")
        return None
    
    logger.info(f"Found {len(files)} result files to combine")
    
    all_rows = []
    for file_path in files:
        try:
            with open(file_path) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                all_rows.extend(data)
            elif isinstance(data, dict):
                all_rows.append(data)
            else:
                logger.warning(f"Unexpected format in {file_path}: {type(data)}")
                
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
    
    if not all_rows:
        logger.warning("No valid results found to combine")
        return None
    
    df = pd.DataFrame(all_rows)
    
    logger.info(f"Combined {len(files)} files with {len(df)} result rows")
    logger.info(f"Columns: {df.columns.tolist()}")
    
    if 'approach' in df.columns:
        approach_counts = df['approach'].value_counts()
        logger.info(f"Approach distribution:\n{approach_counts}")
    
    if 'eval_type' in df.columns:
        eval_counts = df['eval_type'].value_counts()
        logger.info(f"Evaluation types:\n{eval_counts}")
    
    output_path = Path(results_dir) / out_name
    df.to_csv(output_path, index=False)
    logger.info(f"Combined results saved to {output_path}")
    
    return df