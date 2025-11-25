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
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device(device: str = "auto"):
    """Get torch device (cpu or cuda)."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def make_output_dir(output_dir, prefix='ver'):
    """
    make the output directory under output_dir, the directory name follows the format of "output_dir/ver_{version}"
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
    Register class to mapping
    
    Args:
        cls: Dataset class to register
    """
    def inner(cls):
        mapping[cls.__name__] = cls
    
        return cls
    
    return inner


class NumpyEncoder(json.JSONEncoder):
    """Custom encoder for NumPy types in JSON"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_json(data: Dict[str, Any], output_path: Union[str, Path], logger=None):
    """Save dictionary to JSON file with numpy support."""
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
    """Load JSON file if it exists."""
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
