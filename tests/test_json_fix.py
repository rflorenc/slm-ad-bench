#!/usr/bin/env python
"""
Test script to verify JSON serialization fixes for numpy types
"""

import numpy as np
import json
from utils.json_utils import safe_json_dumps, sanitize_for_json

def test_json_serialization():
    """Test that all numpy types can be serialized"""
    
    test_data = {
        "float32_value": np.float32(3.14),
        "float64_value": np.float64(2.71),
        "int32_value": np.int32(42),
        "int64_value": np.int64(12345),
        "nan_value": np.float32(np.nan),
        "inf_value": np.float32(np.inf),
        "array_float32": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "array_float64": np.array([4.0, 5.0, 6.0], dtype=np.float64),
        "bool_value": np.bool_(True),
        "nested_dict": {
            "inner_float32": np.float32(1.23),
            "inner_array": np.array([7.0, 8.0], dtype=np.float32),
            "inner_list": [np.float32(9.0), np.int64(10)]
        },
        "list_with_numpy": [np.float32(11.0), np.int32(12), "normal_string", 13],
        "embedding_metadata": {
            "backend": "local_transformers",
            "batch_size": np.int32(2),
            "max_length": np.int64(128),
            "samples_processed": np.int64(100),
            "embedding_dim": np.int32(768),
            "processing_time": np.float64(42.5),
            "scores": np.array([0.1, 0.2, 0.3], dtype=np.float32)
        }
    }
    
    print("Testing problematic data structure...")
    print("Original data types:")
    for key, value in test_data.items():
        print(f"  {key}: {type(value)} = {value}")
    
    print("\n1. Testing standard json.dumps (should fail)...")
    try:
        json_str = json.dumps(test_data)
        print("UNEXPECTED: Standard json.dumps worked!")
    except TypeError as e:
        print(f"EXPECTED: Standard json.dumps failed: {e}")
    
    print("\n2. Testing sanitize_for_json...")
    try:
        sanitized_data = sanitize_for_json(test_data)
        print("Data sanitized successfully")
        
        print("Sanitized data types:")
        for key, value in sanitized_data.items():
            print(f"  {key}: {type(value)} = {value}")
            
    except Exception as e:
        print(f"FAILED: sanitize_for_json failed: {e}")
        return False
    
    print("\n3. Testing safe_json_dumps...")
    try:
        json_str = safe_json_dumps(test_data)
        print("safe_json_dumps worked!")
        print(f"JSON length: {len(json_str)} characters")
        
        parsed_back = json.loads(json_str)
        print("JSON can be parsed back")
        
    except Exception as e:
        print(f"FAILED: safe_json_dumps failed: {e}")
        return False
    
    print("\n4. Testing typical evaluation result structure...")
    eval_result = {
        "approach": "Granite_3.3-2b-instruct_unsupervised_llm",
        "dataset_type": "unsw-nb15",
        "contamination": np.float64(0.1),
        "anomalies_detected": np.int64(42),
        "total_samples": np.int64(1000),
        "anomaly_ratio": np.float64(0.042),
        "backend": "local_transformers",
        "mode": "unlabeled",
        "nlines": np.int64(1000),
        "embedding_time": np.float64(42.5),
        "embedding_dim": np.int32(768),
        "embedding_metadata": {
            "backend": "local_transformers",
            "batch_size": np.int32(2),
            "max_length": np.int32(128),
            "chunk_size": np.int32(200),
            "samples_processed": np.int64(1000),
            "embedding_dim": np.int32(768)
        },
        "score_mean": np.float32(0.5),
        "score_std": np.float32(0.2),
        "score_separation": np.float64(1.5),
        "num_methods_used": np.int32(3),
        "silhouette_score": np.float64(0.3),
        "davies_bouldin_score": np.float64(0.8)
    }
    
    try:
        json_str = safe_json_dumps(eval_result)
        print("Evaluation result serialization worked!")
        
        parsed_back = json.loads(json_str)
        print("Evaluation result JSON is valid")
        
        return True
        
    except Exception as e:
        print(f"FAILED: Evaluation result serialization failed: {e}")
        return False

if __name__ == "__main__":
    print("=== JSON Serialization Fix Test ===")
    success = test_json_serialization()
    
    if success:
        print("\nJSON serialization should work correctly.")
    else:
        print("\nSome tests failed...")
        exit(1)