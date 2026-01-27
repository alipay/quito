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
from omegaconf import OmegaConf
from transformers import PretrainedConfig

from quito.config.model import ModelConfig
from quito.config.base import BaseConfig
from quito.models.base import BaseModel
from quito.config.auto import AutoConfig

MODEL_MAPPING = BaseModel.REGISTRY

class AutoModel:
    """
    Auto model class that automatically loads the appropriate model.
    
    This class follows the Hugging Face Transformers pattern for automatic
    model loading based on configuration or model name.
    """
    
    @classmethod
    def from_config(cls, config: Union[ModelConfig, PretrainedConfig], local_rank: int, **kwargs) -> BaseModel:
        """
        Create a model instance from a QuitoBench configuration object.
        
        This is the primary entry point for model creation from YAML configuration files.
        The method automatically selects the appropriate model class based on the
        model_name specified in the configuration.
        
        Args:
            config (Union[ModelConfig, PretrainedConfig]): Model configuration object
                containing model architecture parameters and hyperparameters.
            local_rank (int): Local rank for distributed training. Use -1 for single GPU/CPU.
            **kwargs: Additional keyword arguments passed to model initialization.
            
        Returns:
            BaseModel: An instance of the appropriate model class (e.g., PatchTST, 
                Chronos, iTransformer) based on the config.model_name.
                
        Raises:
            ValueError: If the model_name in config is not recognized or not registered.
            
        Example:
            >>> from quito.config import AutoConfig
            >>> config = AutoConfig.from_yaml("configs/pretrain/patchtst/config.yaml")
            >>> model = AutoModel.from_config(config.model_config, local_rank=0)
        """            
        model_class = MODEL_MAPPING.get(config.model_name)
        if model_class is None:
            raise ValueError(f"Unknown model type: {config.model_name}")
        
        model = model_class(config, local_rank, **kwargs)
        model.load(config.checkpoint_path)

        return model

    @classmethod 
    def register(cls, model_type: str, model_class: Type[BaseModel]):
        """
        Register a new model type.
        
        Args:
            model_type: Model type enum
            model_class: Model class
        """
        MODEL_MAPPING[model_type] = model_class
    
    @classmethod
    def list_models(cls) -> List[str]:
        """
        List all available model types.
        
        Returns:
            List of model type names
        """
        return [model_type.value for model_type in MODEL_MAPPING]
