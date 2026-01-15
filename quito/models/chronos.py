"""
Chronos Model Wrapper - Simple and lightweight for time series forecasting.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional, Union
from transformers import PretrainedConfig

from quito.models.base import TimeSeriesModel


class ChronosV2(TimeSeriesModel):
    """
    Lightweight wrapper for Amazon Chronos time series foundation model.
    
    Supports:
    - Zero-shot inference (predict)
    - Fine-tuning/Training (forward)
    - Probabilistic forecasting (predict_prob)
    """
    
    def __init__(self, config: PretrainedConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        
        # Try to load Chronos pipeline
        try:
            from chronos import Chronos2Pipeline
            self.pipeline = Chronos2Pipeline.from_pretrained(
                config.name_or_path,
                device_map=self.device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.model = self.pipeline.model
        
        except ImportError as e:
            raise ImportError(
                "\n" + "="*80 + "\n"
                "Amazon Chronos library not found!\n"
                "\n"
                "To use the Chronos model, install it with:\n"
                "  pip install git+https://github.com/amazon-science/chronos-forecasting.git\n"
                "\n"
                "Or install all optional model dependencies:\n"
                "  pip install -r requirements-optional.txt\n"
                "\n"
                "See requirements-optional.txt for more details on model-specific dependencies.\n"
                + "="*80
            ) from e

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

    def predict(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Generate point forecasts (median).
        """
        with torch.no_grad():
            _, forecast = self.pipeline.predict_quantiles(
                x.permute(0, 2, 1), # chronos expects [N, C, L]
                prediction_length=self.config.forecast_horizon,
            ) # forecast is the median
            # chronos will return a list of tensor of shape [C, forecast_horizon], the length of list is N
            # stack them together along the first dimension
            forecast = torch.stack(forecast, dim=0) # N, C, forecast_horizon
            # choose the median quantile
        #     if hasattr(forecast, 'median'):
        #         pred = forecast.median(dim=1).values
        #     elif isinstance(forecast, torch.Tensor):
        #         pred = forecast.median(dim=1).values
        #     else:
        #         # Fallback for tensor
        #         pred = torch.tensor(forecast).median(dim=1).values
        # return pred.unsqueeze(-1)
        return forecast.permute(0, 2, 1)
    