"""
Auto model classes for QUITO library.

These classes automatically load the appropriate model based on configuration
or model name, following the Hugging Face Transformers pattern.
"""

import os
import json
from typing import Any, Dict, List, Optional, Union, Type
from pathlib import Path
import torch
from torch.utils.data import Dataset

from quito.trainers.base import BaseTrainer
from quito.config.training import TrainerConfig
from quito.models.base import BaseModel


class AutoTrainer:
    """
    Auto model class that automatically loads the appropriate model.
    
    This class follows the Hugging Face Transformers pattern for automatic
    model loading based on configuration or model name.
    """
    
    @classmethod
    def from_config(cls, 
                    model: BaseModel,
                    train_dataset: Optional[Dataset] = None,
                    eval_dataset: Optional[Dataset] = None,
                    config: Optional[TrainerConfig] = None,
                    local_rank: int = -1,
                    global_rank: int = -1,  
                    world_size: int = -1, 
                    use_gpu: int = 1, 
                    **kwargs) -> BaseTrainer:
        """
        Create a trainer instance from configuration.
        
        Automatically selects the appropriate trainer class based on
        config.trainer_name and initializes it with the provided model
        and datasets.
        
        Args:
            model (BaseModel): Model to train.
            train_dataset (Optional[Dataset]): Training dataset. Defaults to None.
            eval_dataset (Optional[Dataset]): Evaluation dataset. Defaults to None.
            config (Optional[TrainerConfig]): Training configuration.
                Must contain trainer_name field. Defaults to None.
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
            global_rank (int, optional): Global rank for distributed training.
                Defaults to -1.
            world_size (int, optional): World size for distributed training.
                Defaults to -1.
            use_gpu (int, optional): Whether to use GPU. Defaults to 1.
            **kwargs: Additional arguments passed to trainer constructor.
        
        Returns:
            BaseTrainer: Initialized trainer instance.
            
        Raises:
            ValueError: If trainer_name is not found in registry.
        """
        trainer_cls = BaseTrainer.REGISTRY[config.trainer_name]
        if trainer_cls is None:
            raise ValueError(f"Unknown trainer type: {config.trainer_name}")
        
        return trainer_cls(model=model, 
                           train_dataset=train_dataset,
                           eval_dataset=eval_dataset,
                           config=config,
                           local_rank=local_rank,
                           global_rank=global_rank,
                           world_size=world_size,
                           use_gpu=use_gpu,
                           **kwargs)
    