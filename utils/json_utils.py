"""
JSON utilities for handling numpy types in serialization
"""

import json
import numpy as np
from typing import Any

class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types"""
    
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def safe_json_dump(data: Any, file_path_or_handle, **kwargs) -> None:
    """Safely dump data to JSON file with numpy type handling"""
    if hasattr(file_path_or_handle, 'write'):
        # It's a file handle
        json.dump(data, file_path_or_handle, cls=NumpyJSONEncoder, **kwargs)
    else:
        # It's a file path
        with open(file_path_or_handle, 'w') as f:
            json.dump(data, f, cls=NumpyJSONEncoder, **kwargs)

def safe_json_dumps(data: Any, **kwargs) -> str:
    """Safely serialize data to JSON string with numpy type handling"""
    return json.dumps(data, cls=NumpyJSONEncoder, **kwargs)

def ensure_json_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON serializable types"""
    if isinstance(obj, dict):
        return {k: ensure_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [ensure_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(ensure_json_serializable(item) for item in obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

# Alias for backward compat
sanitize_for_json = ensure_json_serializable