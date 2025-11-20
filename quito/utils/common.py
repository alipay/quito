import random
import numpy as np
import torch
import os
import logging
from pathlib import Path
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

def save_model(model, path):
    """Stub for saving a model."""
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """Stub for loading a model."""
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model

def create_directory(path):
    """Stub for creating a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)

def get_file_size(path):
    """Stub for getting file size in bytes."""
    return os.path.getsize(path) 

def set_up_env():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    
    return local_rank

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
