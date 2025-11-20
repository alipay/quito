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
from quito.trainers.trainers import Trainer
from quito.config.training import TrainingConfig, TrainerType
from quito.models.base import BaseModel


# Trainer registry mapping trainer types to classes
TRAINER_MAPPING = {
   TrainerType.TRAINER: Trainer
}


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
                    config: Optional[TrainingConfig] = None,
                    local_rank: int = -1,
                    global_rank: int = -1,
                    world_size: int = -1, 
                    **kwargs) -> BaseTrainer:
        """
        Load a model from configuration.
        
        Args:
            config: Model configuration object
            local_rank: Local rank for distributed training
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            Initialized model instance
        """
        trainer_cls = TRAINER_MAPPING.get(config.trainer_name)
        if trainer_cls is None:
            raise ValueError(f"Unknown trainer type: {config.trainer_name}")
        
        return trainer_cls(model=model, 
                           train_dataset=train_dataset,
                           eval_dataset=eval_dataset,
                           config=config,
                           local_rank=local_rank,
                           global_rank=global_rank,
                           world_size=world_size,
                           **kwargs)
    