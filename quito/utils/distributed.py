import os
import sys
import logging
import torch
import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from typing import Callable, Union
from functools import wraps

from quito.utils.common import make_output_dir


class DistributedGroupManager:
    """Context manager for distributed training. Skips DDP if single process or CPU."""
    
    def __init__(self, backend, rank, local_rank, world_size):
        self.backend = backend
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.use_distributed = False

    def __enter__(self):
        # Only initialize distributed if:
        # 1. Multiple processes (world_size > 1)
        # 2. GPU is available
        # 3. Distributed environment variables are set
        if (self.world_size > 1 and 
            torch.cuda.is_available() and 
            os.environ.get("MASTER_ADDR") is not None):
            self.use_distributed = True
            if torch.cuda.is_available():
        torch.cuda.set_device(self.local_rank)
            dist.init_process_group(
                backend=self.backend, 
                rank=self.rank, 
                world_size=self.world_size
            )
            logging.info(f"Initialized distributed training: rank={self.rank}, world_size={self.world_size}")
        else:
            self.use_distributed = False
            if not torch.cuda.is_available():
                logging.info("Running on CPU (no GPU detected)")
            else:
                logging.info(f"Running on single GPU (rank={self.rank}, world_size={self.world_size})")
        
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Clean up the process group when exiting the context."""
        if self.use_distributed and dist.is_initialized():
        dist.destroy_process_group()


def log_rank_zero(message):
    """Log message only from rank 0 in distributed mode, or always in single process mode."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logging.info(message)
    else:
        logging.info(message)


def setup_logging(distributed_rank, save_dir=None, filename="log.txt"):
    """
    Set up logging only for the master (rank 0) process.
    
    Args:
        distributed_rank (int): The rank of the current process.
        save_dir (str, optional): Directory to save the log file.
        filename (str): Name of the log file.
    """
    # Only set up logging for rank 0
    if distributed_rank > 0:
        # For non-zero ranks, you can set a NullHandler to suppress all logs
        logging.getLogger().addHandler(logging.NullHandler())
        return

    # Below configuration only applies to rank 0
    logger = logging.getLogger() # Gets the root logger
    logger.setLevel(logging.DEBUG)
    
    # Check if handlers already exist to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format for log messages
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    # Console Handler (prints to stdout)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (writes to a file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, filename)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
def setup(config_path_or_obj: Union[str, DictConfig], mode):
    """
    Set up the environment with configuration from a YAML file or DictConfig object.
    Automatically defaults to CPU/single process if no distributed environment is detected.
    """
    # Default to single process if env vars not set
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    
    # If no distributed env vars, force single process
    if os.environ.get("MASTER_ADDR") is None:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if isinstance(config_path_or_obj, str):
        config = OmegaConf.load(config_path_or_obj)
        config_source = config_path_or_obj
    else:
        config = config_path_or_obj
        config_source = "config object"

    if rank == 0:
        # make output_dir, only on rank 0 
        output_dir = make_output_dir(config.logging.output_dir)
    else:
        output_dir = None
    
    # config logging
    setup_logging(rank, save_dir=output_dir)
    
    if output_dir:
        config.logging.output_dir = str(output_dir) # Update config with actual path
    
    if rank == 0:
        # Load configuration from YAML file
        logging.info(f"Load configuration from {config_source}")
        # Create output directory
        output_dir = Path(output_dir)    
        # Save final config to output directory
        final_config_path = output_dir / "config.yaml"
        OmegaConf.save(config, final_config_path)
        logging.info(f"Saved {mode} configuration to {final_config_path}")
        
        # Log configuration
        logging.info("=" * 80)
        logging.info(f"{mode} Configuration:")
        logging.info(OmegaConf.to_yaml(config))
        logging.info("=" * 80)
        
        # Log device info
        if torch.cuda.is_available():
            logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logging.info("Running on CPU (no GPU detected)")
    
    return rank, world_size, local_rank, config, output_dir


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if dist.is_initialized():
        if dist.get_rank() == 0:
                return fn(*args, **kwargs)
        else:
            # If not initialized, assume rank 0 (single process)
            return fn(*args, **kwargs)
        return None

    return wrapped_fn
