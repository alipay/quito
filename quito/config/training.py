"""
Training configuration classes for QUITO library.
"""

from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from omegaconf import DictConfig

from quito.config.base import BaseConfig
from quito.config.evaluation import MetricType


class TrainerType(Enum):
    """
    Enumeration of supported trainer types.
    """
    TRAINER = "trainer"


class StrategyType(Enum):
    """
    Enumeration of supported strategy types.
    """
    NO = "no"
    STEPS = "steps"
    EPOCHS = "epochs"


class TaskType(Enum):
    """
    Enumeration of supported task types.
    """
    PRE_TRAIN = 'pretrain'
    FINE_TUNE = 'finetune'
    EVALUATE = 'evaluate'


class ModeType(Enum):
    """
    Enumeration of supported mode types.
    """
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'
    
    
class OptimizerType(Enum):
    """Enumeration of supported optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"
    ADAGRAD = "adagrad"
    RMSprop = "rmsprop"


class SchedulerType(Enum):
    """Enumeration of supported scheduler types."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    STEP = "step"


@dataclass
class TrainingConfig(BaseConfig):
    """
    Configuration for training time series models.
    
    This class defines all parameters needed for training, including
    optimization, scheduling, and distributed training settings.
    """
    trainer_name: TrainerType = TrainerType.TRAINER
    # Basic training parameters
    seed: int = 16
    num_epochs: int = 100
    batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # Loss
    loss: MetricType = MetricType.MSE
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
        self.loss = MetricType(self.loss)
        self.eval_metrics = [MetricType(m) for m in self.eval_metrics]
        self.optimizer = OptimizerType(self.optimizer)
        self.scheduler = SchedulerType(self.scheduler)
        self.trainer_name = TrainerType(self.trainer_name)
        super().__post_init__()
    
    def validate(self):
        """Validate training configuration parameters."""
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
        
        if self.logging_steps <= 0:
            raise ValueError("logging_steps must be positive")
    
    def get_effective_batch_size(self) -> int:
        """Get the effective batch size across all devices and accumulation steps."""
        return self.batch_size * self.gradient_accumulation_steps * self.world_size
    
    def to_huggingface_training_args(self) -> Dict[str, Any]:
        """
        Convert to Hugging Face Transformers TrainingArguments compatible format.
        
        Returns:
            Dictionary compatible with Hugging Face TrainingArguments
        """
        return
    
    @classmethod
    def get_default_config(cls, model_type: str = "general") -> "TrainingConfig":
        """
        Get default training configuration for different model types.
        
        Args:
            model_type: Type of model ("gan", "transformer", "vae", "general")
            
        Returns:
            Default training configuration
        """
        return
            