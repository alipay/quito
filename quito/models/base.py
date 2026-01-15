"""
Base model classes for QUITO library.
"""
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from functools import partial
from transformers import PretrainedConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quito.config.model import ModelConfig
from quito.config.training import TrainerConfig
from quito.utils.common import set_seed, get_device
from quito.metrics import get_metric_fn, cal_score


class BaseModel(nn.Module, ABC):
    """
    Base class for all time series models in QUITO.
    
    This class provides common functionality for model initialization,
    saving/loading, and basic training/evaluation methods.
    """
    REGISTRY = {} # register all subclasses
    
    def __init__(self, config: ModelConfig | PretrainedConfig, local_rank: int):
        """
        Initialize the model with configuration.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.metrics = None  # set up in the trainer
        self.loss_fn = None # set up in the trainer

        # some base attributes setup
        self.seq_len = self.config.seq_len
        self.forecast_horizon = self.config.forecast_horizon
        self.decoder_label_len = self.config.decoder_label_len

        # Device management: handle CPU (-1) and GPU (>=0) cases
        self.device = f'cuda:{local_rank}' if local_rank >= 0 else 'cpu'

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Register the subclass by its name
        BaseModel.REGISTRY[cls.__name__] = cls

    def setup_loss_fn(self, loss_fn, loss_kwargs):
        """
        set up loss function for training purpose
        """
        logging.info(f'Loading loss function {loss_fn}')
        if loss_fn == 'mae':
            self.loss_fn = nn.L1Loss(**loss_kwargs)
        elif loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(**loss_kwargs)
        else:
            raise ValueError(f'Loss function {loss_fn} not supported')

    @abstractmethod
    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model, it is assumed that data is being processed properly.
            
        Returns:
            Output tensor
        """
        pass
    
    @abstractmethod
    def loss(self, x, y, x_mark=None, y_mark=None, **kwargs) -> torch.Tensor:
        """
        Compute the loss for the model.
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def _eval_step(self, x, y, x_mark=None, y_mark=None, **kwargs) -> Tuple[Dict[str, torch.Tensor], Tuple]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Evaluation batch
            
        Returns:
            Dictionary containing loss and other metrics
        """
        pass

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """
        Save the model to a directory.
        
        Args:
            save_directory: Directory to save the model
            **kwargs: Additional arguments
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        
        # Save configuration
        self.config.save(save_directory / "config.json")
        
        # Save model info
        model_info = {
            "model_type": self.__class__.__name__,
            "version": "0.1.0",
            "framework": "pytorch",
        }
        
        with open(save_directory / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs):
        """
        Load a pretrained model from a directory or model hub.
        
        Args:
            pretrained_model_name_or_path: Path to the model or model name
            **kwargs: Additional arguments
            
        Returns:
            Loaded model instance
        """
        model_path = Path(pretrained_model_name_or_path)
        
        # Load configuration
        config = ModelConfig.from_file(model_path / "config.json")
        
        # Create model instance
        model = cls(config, **kwargs)
        
        # Load state dict with proper device handling
        state_dict = torch.load(
            model_path / "pytorch_model.bin",
            map_location=torch.device(model.device)
        )
        model.load_state_dict(state_dict)
        
        return model
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            training_config: Training configuration
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.train()
        
        # Move batch to device
        batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        # Forward pass and loss computation
        loss = self.loss(**batch)
        
        return loss
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Evaluation batch
            
        Returns:
            Dictionary containing loss and other metrics
        """
        self.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            score_dict, y_pred = self._eval_step(**batch)
            
            return score_dict, y_pred

    def generate(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """
        Generate synthetic time series data.
        
        Args:
            batch_size: Number of samples to generate
            **kwargs: Additional generation parameters
            
        Returns:
            Generated time series tensor
        """
        raise NotImplementedError("Generation not implemented for this model")
    
    def predict(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Args:
            x: Input tensor
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction tensor
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x=x, y=y, x_mark=x_mark, y_mark=y_mark, **kwargs)

    def load(self, checkpoint_or_path: Union[str, dict]):
        if isinstance(checkpoint_or_path, str):
            ckpt = torch.load(checkpoint_or_path, map_location='cpu')
        else:
            ckpt = checkpoint_or_path

        if 'model_state_dict' in ckpt:
            self.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            self.load_state_dict(ckpt['state_dict'])
        else:
            raise ValueError(f'No model state dict found in checkpoint {ckpt} !!')
        
        logging.info(f'Load model from checkpoint successfully')
    
    def predict_prob(self, x: torch.Tensor, quantiles: List[float] = [0.1, 0.5, 0.9], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate probabilistic forecasts.
        
        Args:
            x: Input tensor [batch, seq_len, 1]
            quantiles: List of quantiles to compute (e.g. [0.1, 0.5, 0.9])
            
        Returns:
            Dictionary with:
                - 'samples': [batch, num_samples, pred_len, 1]
                - 'quantiles': [batch, len(quantiles), pred_len, 1]
                - 'mean': [batch, pred_len, 1]
        """
        raise NotImplementedError


class TimeSeriesModel(BaseModel):
    """
    Base class for time series models.
    
    This class extends BaseModel with time series specific functionality.
    """
    
    def __init__(self, config: ModelConfig, local_rank=-1):
        super().__init__(config, local_rank)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs):
        raise NotImplementedError

    def loss(self, x: torch.Tensor, y: torch.Tensor, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None,  **kwargs) -> torch.Tensor:
        """
        Compute the loss given batch of inputs
        """
        # construct decoder input for encoder-decoder framework
        x, y_in, x_mark, y_mark = self._construct_model_input(x, y, x_mark, y_mark)
        y_pred = self.forward(x=x, y=y_in, x_mark=x_mark, y_mark=y_mark, **kwargs)
        
        return self.loss_fn(y_pred, y[:, -self.forecast_horizon:, :])

    def _eval_step(self, x: torch.Tensor, y: torch.Tensor, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs):
        """
        Perform a single evaluation step.
        
        Args:
            batch: Evaluation batch
            
        Returns:
            Dictionary containing loss and other metrics
        """
        # construct decoder input for encoder-decoder framework
        x, y_in, x_mark, y_mark = self._construct_model_input(x, y, x_mark, y_mark)
        y_pred = self.predict(x=x, y=y_in, x_mark=x_mark, y_mark=y_mark, **kwargs)
        score_dict = {}
        if isinstance(y_pred, tuple):
            # now it is a tuple of (y_pred_point, y_pred_quantile)
            y_pred_point, y_pred_quantile = y_pred
        else:
            y_pred_point = y_pred
            y_pred_quantile = None
            
        for metric in self.metrics:
            score = cal_score(metric_name=metric, y_pred=y_pred_point, y_true=y[:, -self.forecast_horizon:, :], x_train=x, y_pred_quantile=y_pred_quantile)
            score_dict[metric] = score

        return score_dict, y_pred_point
    
    def _construct_model_input(self, x, y, x_mark, y_mark):
        """ Construct decoder input using y for encoder-decoder framework """            
        dec_in = torch.zeros_like(y[:, -self.forecast_horizon:, :])
        dec_in = torch.cat([y[:, :self.decoder_label_len, :], dec_in], dim=1).float().to(self.device)

        return x, dec_in, x_mark, y_mark
        