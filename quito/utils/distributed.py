import os
import sys
import logging
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable
from functools import wraps

from quito.utils.common import make_output_dir


class DistributedGroupManager:
    def __init__(self, backend, rank, local_rank, world_size):
        self.backend = backend
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size

    def __enter__(self):
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend=self.backend, rank=self.rank, world_size=self.world_size)
        return self  # Optional, for chaining

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Clean up the process group when exiting the context.
        """
        dist.destroy_process_group()

def log_rank_zero(message):
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
        
def setup(config_path, mode):
    """
    Set up the environment with configuration from a YAML file.
    """
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    config = OmegaConf.load(config_path)
    if rank == 0:
        # make output_dir, only on rank 0 
        output_dir = make_output_dir(config.logging.output_dir)
    else:
        output_dir = None
    # config logging
    setup_logging(rank, save_dir=output_dir)
    config.logging.output_dir = output_dir
    
    if rank == 0:
        # Load configuration from YAML file
        logging.info(f"Load configuration from {config_path}")
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
    
    return rank, world_size, local_rank, config, output_dir


def rank_zero_only(fn: Callable) -> Callable:
    """Function that can be used as a decorator to enable a function/method being called only on global rank 0."""

    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if dist.get_rank() == 0:
            return fn(*args, **kwargs)
        return None

    return wrapped_fn