"""
Training configuration classes for QUITO library.
"""

from typing import Optional, List, Union, Dict, Any, ClassVar, Type
from dataclasses import dataclass, field
from enum import Enum
from omegaconf import DictConfig

from quito.config.base import BaseConfig
from quito.config.evaluation import MetricType


class StrategyType(Enum):
    """
    Enumeration of supported strategy types.
    """
    NO = "no"
    STEPS = "steps"
    EPOCHS = "epochs"


class TaskType(Enum):
    """
    Enumeration of supported task types in QuitoBench.
    
    Defines the different types of operations that can be performed:
    - PRE_TRAIN: Pre-training on unlabeled data (out-of-sample training)
    - FINE_TUNE: Fine-tuning on labeled data (in-sample training)
    - EVALUATE: Evaluation on test data
    - TUNE: Hyperparameter tuning on train/validation splits
    """
    PRE_TRAIN = 'pretrain' # perform out-sample traning on TRAIN part of the Pretraining set
    FINE_TUNE = 'finetune' # perform in-sample training on TRAIN part of the Train/test set
    EVALUATE = 'evaluate' # peform evaluation on TEST part of the Train/test set
    TUNE = 'tune' # hyperparameter tuning on TRAIN VALID part of the Train/test set


class ModeType(Enum):
    """
    Enumeration of supported data split modes.
    
    Defines which portion of the dataset to use:
    - TRAIN: Training split
    - VALID: Validation split
    - TEST: Test split
    """
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    
    
class OptimizerType(Enum):
    """
    Enumeration of supported optimizer types.
    
    Defines the optimization algorithms available for training:
    - ADAM: Adaptive Moment Estimation
    - ADAMW: Adam with weight decay
    - SGD: Stochastic Gradient Descent
    - ADAGRAD: Adaptive Gradient Algorithm
    - RMSprop: Root Mean Square Propagation
    """
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    RMSprop = "rmsprop"


class SchedulerType(Enum):
    """
    Enumeration of supported learning rate scheduler types.
    
    Defines the learning rate scheduling strategies:
    - LINEAR: Linear decay
    - COSINE: Cosine annealing
    - COSINE_WITH_RESTARTS: Cosine annealing with restarts
    - POLYNOMIAL: Polynomial decay
    - CONSTANT: Constant learning rate
    - CONSTANT_WITH_WARMUP: Constant with warmup phase
    - STEP: Step-wise decay
    """
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    STEP = "step"


@dataclass
class TrainerConfig(BaseConfig):
    """
    Base configuration class for training time series forecasting models.
    
    This comprehensive configuration class defines all parameters needed for
    training, including optimization, learning rate scheduling, data loading,
    evaluation, checkpointing, logging, and distributed training settings.
    
    The class uses a registry pattern to automatically register subclasses,
    enabling dynamic trainer selection based on configuration.
    
    Attributes:
        REGISTRY (ClassVar[Dict]): Class-level registry of all TrainerConfig subclasses.
        trainer_name (str): Name of the trainer class to use. Defaults to 'NaiveTrainer'.
        
    Example:
        >>> config = TrainerConfig(
        ...     num_epochs=100,
        ...     learning_rate=1e-4,
        ...     batch_size=32,
        ...     optimizer=OptimizerType.ADAMW
        ... )
    """
    REGISTRY: ClassVar[Dict[str, Type["TrainerConfig"]]] = {}
    trainer_name: str = 'NaiveTrainer'
    # Basic training parameters
    seed: int = 16
    num_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Loss
    loss: str = 'mse'
    loss_kwargs: dict = field(default_factory={})
    
    # Learning rate and optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    max_grad_norm: float = 1.0
    optimizer: OptimizerType = OptimizerType.ADAMW
    optimizer_kwargs: dict = field(default_factory=dict) 
    
    # Scheduler parameters
    scheduler: SchedulerType = SchedulerType.LINEAR
    scheduler_kwargs: dict = field(default_factory=dict)
    warmup_steps: int = 0
    warmup_ratio: float = 0.0
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    shuffle: bool = True
    
    # Evaluation
    eval_batch_size: int = 32
    eval_metrics: list = field(default_factory=lambda: [MetricType.MSE, MetricType.MAE])
    eval_steps: Optional[int] = 10000
    eval_epochs: Optional[int] = 1
    save_eval_results_top_k: int = 0 # wheather to save the last k eval step results, the inputs, predictions and labels during evaluation

    # Checkpointing
    enable_checkpoints: bool = True
    save_last_k: int = 10
    save_steps: Optional[int] = 10000
    save_epochs: Optional[int] = 1
    checkpoint_path: str = None
    
    # Logging
    output_dir: str = 'outputs'
    logging_steps: Optional[int] = 100
    logging_epochs: Optional[int] = 1
    sync_loss: bool = False # If true, training loss will be synced across devices
    sync_score: bool = True # If true, validation score will be synced across devices
    
    # Distributed training
    local_rank: int = -1
    ddp_backend: str = "nccl"
    ddp_find_unused_parameters: bool = False
    world_size: int = 1
    global_rank: int = -1
    
    # Mixed precision
    fp16: bool = False

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_patience: Optional[int] = None
    early_stopping_threshold: float = 0.0
    es_metric: str = "mse"  # Metric to monitor for early stopping
    greater_is_better: bool = False  # is_greater_better for the metric
    
    def __post_init__(self):
        # convert to corresponding Enum constants
        self.es_metric = MetricType(self.es_metric)
        self.eval_metrics = [MetricType(m) for m in self.eval_metrics]
        self.optimizer = OptimizerType(self.optimizer)
        self.scheduler = SchedulerType(self.scheduler)
        self.trainer_name = self.trainer_name
        super().__post_init__()
    
    def validate(self):
        """
        Validate training configuration parameters.
        
        Checks that all critical parameters are within valid ranges and
        that the configuration is consistent.
        
        Raises:
            ValueError: If any parameter is invalid (e.g., negative values,
                ratios outside [0, 1], etc.).
        """
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        
        if not 0 <= self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be between 0 and 1")
        
        if self.num_workers < 0:
            raise ValueError("dataloader_num_workers must be non-negative")
        
        if self.logging_steps < 0:
            raise ValueError("logging_steps must be positive")
    
    def get_effective_batch_size(self) -> int:
        """
        Calculate the effective batch size across all devices and gradient accumulation.
        
        The effective batch size accounts for:
        - Base batch size per device
        - Gradient accumulation steps (simulating larger batches)
        - Number of distributed training processes (world_size)
        
        Returns:
            int: Effective batch size = batch_size * gradient_accumulation_steps * world_size.
            
        Example:
            >>> config.batch_size = 32
            >>> config.gradient_accumulation_steps = 4
            >>> config.world_size = 2
            >>> effective = config.get_effective_batch_size()
            >>> # Returns: 256 (32 * 4 * 2)
        """
        return self.batch_size * self.gradient_accumulation_steps * self.world_size
    
    def to_huggingface_training_args(self) -> Dict[str, Any]:
        """
        Convert to Hugging Face Transformers TrainingArguments compatible format.
        
        Converts QuitoBench training configuration to a dictionary format
        compatible with Hugging Face's TrainingArguments class, enabling
        interoperability with Hugging Face training utilities.
        
        Returns:
            Dict[str, Any]: Dictionary with keys matching Hugging Face
                TrainingArguments parameters.
                
        Note:
            This method is a stub and should be implemented for full
            Hugging Face integration.
        """
        return
    
    @classmethod
    def get_default_config(cls, model_type: str = "general") -> "TrainerConfig":
        """
        Get default training configuration for different model types.
        
        Provides sensible default configurations tailored to different
        model architectures (GANs, transformers, VAEs, etc.).
        
        Args:
            model_type (str, optional): Type of model. Options:
                - "gan": Generative Adversarial Network defaults
                - "transformer": Transformer model defaults
                - "vae": Variational Autoencoder defaults
                - "general": General purpose defaults
                Defaults to "general".
        
        Returns:
            TrainerConfig: Default training configuration instance.
            
        Note:
            This method is a stub and should be implemented with
            model-specific defaults.
        """
        return

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        TrainerConfig.REGISTRY[cls.__name__] = cls

@dataclass
class NaiveTrainerTrainerConfig(TrainerConfig):
    """
    Configuration for the NaiveTrainer.
    
    A simple trainer configuration that provides basic training functionality
    without advanced features. Automatically registered in TrainerConfig.REGISTRY.
    """
    pass