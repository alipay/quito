"""
Chronos Model Wrapper - Simple and lightweight for time series forecasting.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional, Union
from transformers import PretrainedConfig
from einops import rearrange

from quito.models.base import TimeSeriesModel


class TimesFMV2p5(TimeSeriesModel):
    """
    Lightweight wrapper for Google TimesFM 2.5-200B time series foundation model.
    
    TimesFM is a large-scale pre-trained foundation model for time series forecasting
    that supports zero-shot inference across diverse domains. The 2.5-200B variant
    is a 200M parameter model.
    
    Installation:
        Install from: https://github.com/google-research/timesfm
    
    Supports:
    - Zero-shot inference (predict)
    - Fine-tuning/Training (forward) - Not implemented
    - Probabilistic forecasting (predict_prob) - Not implemented
    
    Reference:
        Google Research TimesFM: Time Series Foundation Model
    """
    
    def __init__(self, config: PretrainedConfig, local_rank: int = -1):
        """
        Initialize the TimesFM model.
        
        Args:
            config (PretrainedConfig): Model configuration containing:
                - name_or_path: Model name or path
                - seq_len: Maximum context length
                - forecast_horizon: Forecast horizon
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
                
        Raises:
            ImportError: If timesfm library is not installed.
        """
        super().__init__(config, local_rank)
        
        # Try to load Chronos pipeline
        try:
            import timesfm
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(config.name_or_path)
            self.model = model
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=config.seq_len,
                    max_horizon=config.forecast_horizon,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                    per_core_batch_size=1024,
            )
            )
        
        except ImportError as e:
            raise ImportError(
                "\n" + "="*80 + "\n"
                "TimesFM library not found!\n"
                + "="*80
            )
    
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
            score_dict, y_pred = self._eval_step(**batch)
    
            return score_dict, y_pred

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
        Generate point forecasts using TimesFM zero-shot inference.
        
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
            # for timesfm, the input is a list of np.array, we need to flatten the tensor
            N, L, C = x.shape
            x = rearrange(x, 'n l c -> (n c) l ') # n, l, c -> n, l*c
            x = list(x.numpy())

            point_forecast, _ = self.model.forecast(
                inputs=x,
                horizon=self.config.forecast_horizon
            ) # forecast is the median
            # chronos will return a list of tensor of shape [C, forecast_horizon], the length of list is N
            # stack them together along the first dimension
            forecast = rearrange(point_forecast, '(n c) l -> n l c', n=N, c=C)
            forecast = torch.from_numpy(forecast).float()
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