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
from quito.config.training import TaskType


class DistributedGroupManager:
    """
    Context manager for distributed training with automatic fallback to single-process mode.
    
    Manages PyTorch distributed process group initialization and cleanup. Automatically
    detects whether to use distributed training based on world size, GPU availability,
    and environment variables. Falls back to single-process mode gracefully.
    
    Args:
        backend (str): Distributed backend (e.g., 'nccl', 'gloo').
        rank (int): Global rank of the current process.
        local_rank (int): Local rank on the current node.
        world_size (int): Total number of processes.
    
    Attributes:
        use_distributed (bool): Whether distributed mode is actually being used.
        
    Example:
        >>> with DistributedGroupManager('nccl', rank=0, local_rank=0, world_size=4):
        ...     # Distributed training code here
        ...     model = DistributedDataParallel(model)
    """

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
    """
    Log a message only from rank 0 process in distributed training.
    
    In distributed mode, only the rank 0 process logs the message to avoid
    duplicate logging. In single-process mode, always logs the message.
    
    Args:
        message (str): Message to log at INFO level.
        
    Example:
        >>> log_rank_zero("Training started")  # Only rank 0 logs this
        >>> # Single GPU or rank 0: INFO: Training started
        >>> # Other ranks: (no output)
    """
    if dist.is_initialized():
        if dist.get_rank() == 0:
            logging.info(message)
    else:
        logging.info(message)


def setup_logging(distributed_rank, save_dir=None, filename="log.txt"):
    """
    Configure logging for distributed training (rank 0 only).
    
    Sets up both console and file logging for the master process (rank 0).
    Non-master processes are silenced to avoid duplicate logs. Configures
    formatters and handlers for structured logging output.

    Args:
        distributed_rank (int): The rank of the current process. Only rank 0
            will have active logging configured.
        save_dir (str, optional): Directory to save the log file. If None,
            only console logging is configured. Defaults to None.
        filename (str, optional): Name of the log file. Defaults to "log.txt".
        
    Example:
        >>> setup_logging(rank=0, save_dir="outputs/exp1")
        >>> # Logs will be written to outputs/exp1/log.txt and console
    """
    # Only set up logging for rank 0
    if distributed_rank > 0:
        # For non-zero ranks, you can set a NullHandler to suppress all logs
        logging.getLogger().addHandler(logging.NullHandler())
        return

    # Below configuration only applies to rank 0
    logger = logging.getLogger()  # Gets the root logger
    logger.setLevel(logging.INFO)

    # Check if handlers already exist to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Format for log messages
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")

    # Console Handler (prints to stdout)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File Handler (writes to a file)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        log_file = os.path.join(save_dir, filename)
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)


def setup_train(config_path_or_obj: Union[str, DictConfig], mode):
    """
    Set up training environment with configuration and distributed settings.
    
    Initializes the training environment by loading configuration, detecting
    distributed training settings from environment variables, creating output
    directories, and configuring logging. Automatically falls back to single-process
    mode if no distributed environment is detected.
    
    Args:
        config_path_or_obj (Union[str, DictConfig]): Path to YAML configuration file
            or an OmegaConf DictConfig object.
        mode: Task mode (e.g., TaskType.PRE_TRAIN or TaskType.FINE_TUNE).
    
    Returns:
        tuple: A tuple containing:
            - rank (int): Global rank of current process
            - world_size (int): Total number of processes
            - local_rank (int): Local rank on current node
            - config (DictConfig): Loaded configuration object
            - output_dir (Path): Path to output directory
            
    Example:
        >>> rank, world_size, local_rank, config, output_dir = setup_train(
        ...     "configs/pretrain/patchtst/config.yaml",
        ...     TaskType.PRE_TRAIN
        ... )
    """
    # Default to single process if env vars not set
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    # If no distributed env vars, force single process
    if os.environ.get("MASTER_ADDR") is None:
        if torch.cuda.is_available():
            rank = 0
            world_size = 1
            local_rank = 0
        else:
            rank = -1
            world_size = -1
            local_rank = -1

    if isinstance(config_path_or_obj, str):
        config = OmegaConf.load(config_path_or_obj)
        config_source = config_path_or_obj
    else:
        config = config_path_or_obj
        config_source = "config object"

    if rank in [0, -1]:
        # make output_dir, only on rank 0 or cpu mode
        output_dir = make_output_dir(os.path.join(config.logging.output_dir, mode.name))
    else:
        output_dir = None

    # config logging
    setup_logging(rank, save_dir=output_dir)

    if output_dir:
        config.logging.output_dir = str(output_dir)  # Update config with actual path

    if rank in [0, -1]:
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


def setup_evaluation(config_path_or_obj: Union[str, DictConfig], mode):
    """
    Set up evaluation environment with configuration and logging.
    
    Initializes the evaluation environment by loading configuration, creating
    output directories, and configuring logging. Similar to setup_train but
    without distributed training setup since evaluation typically runs on
    a single process.
    
    Args:
        config_path_or_obj (Union[str, DictConfig]): Path to YAML configuration
            file or an OmegaConf DictConfig object.
        mode: Task mode (typically TaskType.EVALUATE).
    
    Returns:
        tuple: A tuple containing:
            - config (DictConfig): Loaded configuration object
            - output_dir (Path): Path to output directory
            
    Example:
        >>> config, output_dir = setup_evaluation(
        ...     "configs/evaluate/chronos/config.yaml",
        ...     TaskType.EVALUATE
        ... )
    """
    if isinstance(config_path_or_obj, str):
        config = OmegaConf.load(config_path_or_obj)
        config_source = config_path_or_obj
    else:
        config = config_path_or_obj
        config_source = "config object"

    output_dir = make_output_dir(os.path.join(config.logging.output_dir, mode.name))
    config.logging.output_dir = str(output_dir)  # Update config with actual path
    setup_logging(0, save_dir=output_dir)
    # Load configuration from YAML file
    logging.info(f"Load configuration from {config_source}")
    # Create output directory
    output_dir = Path(output_dir)
    # Save final config to output directory
    final_config_path = output_dir / "config.yaml"
    OmegaConf.save(config, final_config_path)
    logging.info(f"Saved configuration to {final_config_path}")

    return config, output_dir


def setup_tuning(config_path_or_obj: Union[str, DictConfig], mode):
    """
    Set up hyperparameter tuning environment with configuration and logging.
    
    Initializes the tuning environment by loading configuration, creating
    output directories, and configuring logging. Used for hyperparameter
    optimization tasks (e.g., with Ray Tune).
    
    Args:
        config_path_or_obj (Union[str, DictConfig]): Path to YAML configuration
            file or an OmegaConf DictConfig object.
        mode: Task mode (typically TaskType.TUNE).
    
    Returns:
        tuple: A tuple containing:
            - config (DictConfig): Loaded configuration object
            - output_dir (Path): Path to output directory
            
    Example:
        >>> config, output_dir = setup_tuning(
        ...     "configs/tune/patchtst/config.yaml",
        ...     TaskType.TUNE
        ... )
    """
    if isinstance(config_path_or_obj, str):
        config = OmegaConf.load(config_path_or_obj)
        config_source = config_path_or_obj
    else:
        config = config_path_or_obj
        config_source = "config object"
    output_dir = os.path.abspath(config.logging.output_dir)
    output_dir = make_output_dir(os.path.join(output_dir, mode.name))
    config.logging.output_dir = str(output_dir)  # Update config with actual path
    # Load configuration from YAML file
    logging.info(f"Load configuration from {config_source}")
    # Create output directory
    output_dir = Path(output_dir)
    # Save final config to output directory
    final_config_path = output_dir / "config.yaml"
    OmegaConf.save(config, final_config_path)
    logging.info(f"Saved configuration to {final_config_path}")

    return config, output_dir


def setup(config_path_or_obj: Union[str, DictConfig], mode):
    """
    Unified setup function that routes to appropriate setup based on task mode.
    
    Automatically selects the correct setup function (setup_train, setup_evaluation,
    or setup_tuning) based on the task mode. This provides a single entry point
    for environment initialization across all task types.
    
    Args:
        config_path_or_obj (Union[str, DictConfig]): Path to YAML configuration
            file or an OmegaConf DictConfig object.
        mode (TaskType): Task type determining which setup function to use:
            - TaskType.EVALUATE: Uses setup_evaluation
            - TaskType.FINE_TUNE or TaskType.PRE_TRAIN: Uses setup_train
            - TaskType.TUNE: Uses setup_tuning
    
    Returns:
        tuple: Return value depends on task mode:
            - For EVALUATE/TUNE: (config, output_dir)
            - For FINE_TUNE/PRE_TRAIN: (rank, world_size, local_rank, config, output_dir)
            
    Example:
        >>> # For training
        >>> rank, world_size, local_rank, config, output_dir = setup(
        ...     "configs/pretrain/patchtst/config.yaml",
        ...     TaskType.PRE_TRAIN
        ... )
        >>> # For evaluation
        >>> config, output_dir = setup(
        ...     "configs/evaluate/chronos/config.yaml",
        ...     TaskType.EVALUATE
        ... )
    """
    if mode == TaskType.EVALUATE:
        return setup_evaluation(config_path_or_obj, mode)
    elif (mode == TaskType.FINE_TUNE) or (mode == TaskType.PRE_TRAIN):
        return setup_train(config_path_or_obj, mode)
    elif mode == TaskType.TUNE:
        return setup_tuning(config_path_or_obj, mode)


def rank_zero_only(fn: Callable) -> Callable:
    """
    Decorator to execute a function only on rank 0 in distributed training.
    
    Wraps a function so that it only executes on the rank 0 process in distributed
    mode, preventing redundant operations. In single-process mode, the function
    always executes. Non-rank-0 processes return None.
    
    Args:
        fn (Callable): Function to wrap with rank-zero-only execution.
    
    Returns:
        Callable: Wrapped function that only executes on rank 0.
        
    Example:
        >>> @rank_zero_only
        ... def save_checkpoint(model, path):
        ...     torch.save(model.state_dict(), path)
        >>> 
        >>> save_checkpoint(model, "checkpoint.pt")  # Only rank 0 saves
    """

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
