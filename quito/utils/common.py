import random
import numpy as np
import torch
import os
import logging
import json
from pathlib import Path
from typing import Dict, Any, Union
from omegaconf import OmegaConf, DictConfig
import torch.distributed as dist
from typing import Callable
from functools import wraps


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility across all random number generators.
    
    This function sets the seed for Python's random module, NumPy, and PyTorch
    (both CPU and CUDA) to ensure reproducible results across runs.
    
    Args:
        seed (int, optional): Random seed value. Defaults to 42.
        
    Example:
        >>> set_seed(42)  # All random operations will now be deterministic
        >>> # Subsequent model training will produce same results
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device: str = "auto"):
    """
    Get PyTorch device for computation.
    
    Automatically selects the appropriate device (CPU or CUDA) based on availability,
    or uses a specified device string.
    
    Args:
        device (str, optional): Device specification. Can be:
            - "auto": Automatically select CUDA if available, otherwise CPU
            - "cpu": Force CPU usage
            - "cuda": Use default CUDA device
            - "cuda:0", "cuda:1", etc.: Use specific CUDA device
            Defaults to "auto".
    
    Returns:
        torch.device: PyTorch device object for tensor operations.
        
    Example:
        >>> device = get_device("auto")
        >>> model = model.to(device)
        >>> tensor = torch.randn(10, 10).to(device)
    """
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def make_output_dir(output_dir, prefix='ver'):
    """
    Create a versioned output directory to avoid overwriting existing results.
    
    Creates a new directory with an auto-incremented version number. The function
    searches for existing directories matching the pattern "{prefix}_{version}"
    and creates a new one with the next available version number.
    
    Args:
        output_dir (str): Base output directory path.
        prefix (str, optional): Prefix for versioned subdirectory names. 
            Defaults to 'ver'.
    
    Returns:
        str: Path to the newly created versioned directory (e.g., "output_dir/ver_0").
        
    Example:
        >>> output_path = make_output_dir("experiments/run1")
        >>> # Creates: experiments/run1/ver_0/
        >>> output_path2 = make_output_dir("experiments/run1")
        >>> # Creates: experiments/run1/ver_1/
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    version = 0
    while os.path.exists(os.path.join(output_dir, f'{prefix}_{version}')):
        version += 1
    
    output_dir = os.path.join(output_dir, f'ver_{version}')
    os.makedirs(output_dir)
    
    return output_dir


def register_to_mapping(mapping):
    """
    Decorator factory for registering classes to a mapping dictionary.
    
    This decorator registers a class to a provided mapping dictionary using the
    class name as the key. Commonly used for auto-discovery and registration of
    dataset, model, or trainer classes.
    
    Args:
        mapping (dict): Dictionary to register the class into.
    
    Returns:
        Callable: Decorator function that registers a class and returns it unchanged.
        
    Example:
        >>> MODEL_MAPPING = {}
        >>> @register_to_mapping(MODEL_MAPPING)
        ... class MyModel:
        ...     pass
        >>> print(MODEL_MAPPING)  # {'MyModel': <class 'MyModel'>}
    """
    def inner(cls):
        mapping[cls.__name__] = cls
    
        return cls
    
    return inner


class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles NumPy data types.
    
    This encoder extends json.JSONEncoder to properly serialize NumPy integers,
    floats, and arrays to their Python equivalents, enabling JSON serialization
    of data structures containing NumPy objects.
    
    Supported conversions:
        - np.integer types → Python int
        - np.floating types → Python float
        - np.ndarray → Python list
    
    Example:
        >>> import numpy as np
        >>> data = {'array': np.array([1, 2, 3]), 'value': np.float64(3.14)}
        >>> json.dumps(data, cls=NumpyEncoder)
        '{"array": [1, 2, 3], "value": 3.14}'
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_json(data: Dict[str, Any], output_path: Union[str, Path], logger=None):
    """
    Save dictionary to JSON file with NumPy type support.
    
    Serializes a dictionary to a JSON file, automatically handling NumPy types
    using the NumpyEncoder. Creates parent directories if they don't exist.
    
    Args:
        data (Dict[str, Any]): Dictionary to save. May contain NumPy types.
        output_path (Union[str, Path]): Path where JSON file will be saved.
        logger (logging.Logger, optional): Logger for status messages. 
            If None, prints to stdout. Defaults to None.
    
    Example:
        >>> import numpy as np
        >>> results = {'accuracy': np.float64(0.95), 'predictions': np.array([1, 0, 1])}
        >>> save_json(results, 'results.json')
    """
    output_path = Path(output_path)
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, cls=NumpyEncoder, indent=2)
        if logger:
            logger.info(f"Saved results to {output_path}")
    except Exception as e:
        if logger:
            logger.error(f"Failed to save results: {e}")
        else:
            print(f"Failed to save results: {e}")


def load_json(output_path: Union[str, Path], logger=None) -> Dict[str, Any]:
    """
    Load JSON file from disk if it exists.
    
    Attempts to load and parse a JSON file. Returns empty dictionary if the
    file doesn't exist or if parsing fails.
    
    Args:
        output_path (Union[str, Path]): Path to JSON file to load.
        logger (logging.Logger, optional): Logger for warning messages.
            If None, prints to stdout. Defaults to None.
    
    Returns:
        Dict[str, Any]: Loaded JSON data as dictionary. Returns empty dict
            if file doesn't exist or parsing fails.
            
    Example:
        >>> data = load_json('results.json')
        >>> if data:
        ...     print(f"Loaded {len(data)} keys")
    """
    output_path = Path(output_path)
    if output_path.exists():
        try:
            with open(output_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            msg = f"Could not load existing results from {output_path}: {e}"
            if logger:
                logger.warning(msg)
            else:
                print(msg)
            return {}
    return {}

def deep_update(original, update_with):
    """
    Recursively update a nested dictionary with values from another dictionary.
    
    Performs a deep merge of two dictionaries, recursively updating nested
    dictionary values. Non-dict values are overwritten.
    
    Args:
        original (dict): Original dictionary to be updated (modified in-place).
        update_with (dict): Dictionary containing updates to merge.
    
    Returns:
        dict: The updated original dictionary (same object, modified in-place).
        
    Note:
        The original dictionary is modified in-place.
        
    Example:
        >>> original = {'a': {'b': 1, 'c': 2}, 'd': 3}
        >>> updates = {'a': {'c': 3, 'e': 4}, 'f': 5}
        >>> result = deep_update(original, updates)
        >>> print(result)  # {'a': {'b': 1, 'c': 3, 'e': 4}, 'd': 3, 'f': 5}
    """
    for k, v in update_with.items():
        original[k].update(update_with[k])
            
    return original