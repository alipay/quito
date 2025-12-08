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

from quito.models.patchtst import PatchTST
from quito.models.dlinear import DLinear
from quito.models.chronos import ChronosModel
from quito.models.moriai import MoriaiModel
from quito.models.huggingface import HuggingFaceModel
from quito.models.tstransformer import TSTransformer 
from quito.config.model import ModelType, ModelConfig
from quito.config.base import BaseConfig
from quito.models.base import BaseModel
from quito.config.auto import AutoConfig


# Model registry mapping model types to classes
MODEL_MAPPING = {
    ModelType.PATCHTST: PatchTST,
    ModelType.DLINEAR: DLinear,
    ModelType.CHRONOS: ChronosModel,
    ModelType.MORIAI: MoriaiModel,
    ModelType.HUGGINGFACE: HuggingFaceModel,
    ModelType.TSTRANSFORMER: TSTransformer
}

class AutoModel:
    """
    Auto model class that automatically loads the appropriate model.
    
    This class follows the Hugging Face Transformers pattern for automatic
    model loading based on configuration or model name.
    """
    
    @classmethod
    def from_config(cls, config: ModelConfig, local_rank: int, **kwargs) -> BaseModel:
        """
        Create a model from configuration.
        
        Args:
            config: Model configuration
            **kwargs: Additional arguments
            
        Returns:
            Model instance
        """
        model_class = MODEL_MAPPING.get(config.model_name)
        if model_class is None:
            raise ValueError(f"Unknown model type: {config.model_name}")
        
        return model_class(config, local_rank, **kwargs)
    
    @classmethod
    def from_pretrained(cls, local_config_path: str = None, pretrained_model_name_or_path: str = None, local_rank=-1, rank=-1, world_size=-1, **kwargs) -> BaseModel:
        """
        Load a pretrained model from
        1. directory or model hub (Huggingface).
        2. a path to yaml config
        
        Returns:
            Loaded model instance
        """
        if local_config_path:
            # load from local config file, need to first load config then init model.
            config = OmegaConf.load(local_config_path)
            _, model_config, training_config = AutoConfig.from_config(config, local_rank=local_rank, rank=rank, world_size=world_size)
            model = cls.from_config(model_config, local_rank=local_rank, **kwargs)
            checkpoint_path = training_config.checkpoint_path
            model.load(checkpoint_path)

            return model
        else:
            # TODO: load from prefrained model hub or a huggingface local directory
            raise NotImplementedError
    
    @classmethod
    def _from_pretrained_remote(cls, model_name: str, **kwargs) -> BaseModel:
        """
        Load a model from remote source (model hub, etc.).
        
        Args:
            model_name: Model name
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        # TODO: we need able to load from huggingface repo
        pass
    
    @classmethod
    def _from_local_checkpoint(cls, checkpoint_path: str, **kwargs):
        
        pass

    @classmethod
    def _from_pretrained_local(cls, model_path: str, **kwargs):
        """
        Load from huggingface local file path.
        """
        # TODO: we need able to load from huggingface local file path
        pass

    @classmethod
    def register(cls, model_type: ModelType, model_class: Type[BaseModel]):
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
