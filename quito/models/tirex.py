"""
Chronos Model Wrapper - Simple and lightweight for time series forecasting.
"""
import os
import logging
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional, Union
from transformers import PretrainedConfig
from einops import rearrange

from quito.models.base import TimeSeriesModel


class TiRex(TimeSeriesModel):
    """
    Lightweight wrapper for NX-AI TiRex time series foundation model.
    
    TiRex is a pre-trained foundation model for time series forecasting that
    supports zero-shot inference without fine-tuning. It's designed for
    efficient forecasting across diverse time series domains.
    
    Installation:
        Requires Python >= 3.11. Install from:
        https://github.com/NX-AI/tirex.git
        
    Supports:
    - Zero-shot inference (predict)
    - Fine-tuning/Training (forward) - Not implemented
    - Probabilistic forecasting (predict_prob) - Not implemented
    
    Reference:
        NX-AI TiRex: Time Series Foundation Model
    """
    
    def __init__(self, config: PretrainedConfig, local_rank: int = -1):
        """
        Initialize the TiRex model.
        
        Args:
            config (PretrainedConfig): Model configuration containing
                pretrained_model_name_or_path and forecast_horizon.
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
                
        Raises:
            ImportError: If tirex library is not installed.
        """
        super().__init__(config, local_rank)
        
        # Try to load Chronos pipeline
        try:
            from tirex import TiRexZero
            model_path = config.pretrained_model_name_or_path
            # tirex expects a path directly to the model.ckpt if use locally.
            if os.path.exists(model_path):
                if not model_path.endswith('model.ckpt'):
                    model_path = os.path.join(model_path, 'model.ckpt')

            self.model = TiRexZero.from_pretrained(
                model_path,
                backend='torch'
            )
        
        except ImportError as e:
            raise ImportError(
                "\n" + "="*80 + "\n"
                "Tirex library not found!\n"
                + "="*80
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: Context time series [batch, seq_len, 1]
            y: Target time series [batch, pred_len, 1]
            **kwargs: Additional arguments (filtered to avoid conflicts)
            
        Returns:
            Loss tensor for training
        """
        raise NotImplementedError("ChronosV2 model does not support training directly.")
    

    def predict(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Generate point forecasts using TiRex zero-shot inference.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor (not used). Defaults to None.
            x_mark (torch.Tensor, optional): Time features (not used). Defaults to None.
            y_mark (torch.Tensor, optional): Time features (not used). Defaults to None.
            **kwargs: Additional arguments (not used).
        
        Returns:
            torch.Tensor: Point forecasts (median) of shape
                (batch_size, forecast_horizon, n_features).
        """
        with torch.no_grad():
            # tirex only accepts N, L
            N, L, C = x.shape
            x = rearrange(x, 'n l c -> (n c) l')
            _, point_forecast = self.model.forecast(
                x,
                prediction_length=self.config.forecast_horizon,
                output_type='torch',
                batch_size=N * C,
            )
            # forecast is the median
            # chronos will return a list of tensor of shape [C, forecast_horizon], the length of list is N
            # stack them together along the first dimension
            forecast = rearrange(point_forecast, '(n c) l -> n l c', n=N, c=C)
            # forecast = torch.from_numpy(forecast).float()
            # choose the median quantile
        #     if hasattr(forecast, 'median'):
        #         pred = forecast.median(dim=1).values
        #     elif isinstance(forecast, torch.Tensor):
        #         pred = forecast.median(dim=1).values
        #     else:
        #         # Fallback for tensor
        #         pred = torch.tensor(forecast).median(dim=1).values
            
        # return pred.unsqueeze(-1)
        return forecast