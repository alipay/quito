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
    
    Chronos is a pre-trained foundation model for time series forecasting that
    supports zero-shot inference without fine-tuning. This wrapper integrates
    Chronos into the QuitoBench framework for evaluation and benchmarking.
    
    Supports:
    - Zero-shot inference (predict): Generate forecasts without training
    - Probabilistic forecasting: Quantile predictions for uncertainty estimation
    - Evaluation: Standard QuitoBench evaluation metrics
    
    Note:
        Chronos does not support training/fine-tuning in this implementation.
        It is designed for zero-shot evaluation on new datasets.
        
    Requirements:
        - chronos-forecasting package must be installed
        - Pre-trained Chronos model weights (loaded from HuggingFace)
        
    Example:
        >>> config = ChronosModelConfig(
        ...     pretrained_model_name_or_path="amazon/chronos-t5-base",
        ...     seq_len=512,
        ...     forecast_horizon=96
        ... )
        >>> model = ChronosV2(config, local_rank=0)
        >>> model.load("./models/chronos")
        >>> predictions = model.predict(input_tensor)
    """
    
    def __init__(self, config: PretrainedConfig, local_rank: int = -1):
        """
        Initialize the Chronos model wrapper.
        
        Loads the Chronos pipeline from the specified pretrained model path
        and configures it for the appropriate device (CPU or GPU).
        
        Args:
            config (PretrainedConfig): Model configuration containing:
                - name_or_path: Path to pretrained Chronos model (HuggingFace ID or local path)
                - forecast_horizon: Prediction horizon length
                - seq_len: Input sequence length (for compatibility)
            local_rank (int, optional): Local rank for distributed training.
                Use -1 for CPU, >=0 for GPU. Defaults to -1.
                
        Raises:
            ImportError: If chronos-forecasting package is not installed.
            FileNotFoundError: If the pretrained model cannot be loaded.
            
        Note:
            The model uses bfloat16 precision on GPU and float32 on CPU for
            optimal performance and memory usage.
        """
        super().__init__(config, local_rank)
        self.model = None
        self.pipeline = None

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for training (not supported for Chronos).
        
        Chronos is a pre-trained foundation model designed for zero-shot inference.
        Training/fine-tuning is not supported in this implementation.
        
        Args:
            x (torch.Tensor): Context time series of shape (batch, seq_len, n_features).
            y (torch.Tensor, optional): Target time series of shape (batch, pred_len, n_features).
                Defaults to None.
            **kwargs: Additional arguments (filtered to avoid conflicts).
        
        Returns:
            torch.Tensor: Loss tensor (not implemented).
            
        Raises:
            NotImplementedError: Always raised, as Chronos does not support training.
            
        Note:
            For inference, use the predict() method instead.
        """
        raise NotImplementedError("ChronosV2 model does not support training directly.")
    
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a single evaluation step on a batch of data.
        
        Evaluates the Chronos model on a batch, computing predictions and
        evaluation metrics. Sets the model to evaluation mode and disables
        gradient computation for efficiency.
        
        Args:
            batch (Dict[str, torch.Tensor]): Evaluation batch dictionary containing:
                - 'x': Input tensor of shape (batch_size, seq_len, n_features)
                - 'y': Target tensor of shape (batch_size, forecast_horizon, n_features)
                - Optional: 'x_mark', 'y_mark' (time features, ignored by Chronos)
        
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
                - score_dict: Dictionary mapping metric names to computed values
                - y_pred: Predicted tensor of shape (batch_size, forecast_horizon, n_features)
        """
        self.eval()
        
        with torch.no_grad():
            score_dict, y_pred = self._eval_step(**batch)
    
            return score_dict, y_pred

    def predict(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Generate point forecasts using Chronos zero-shot inference.
        
        Performs zero-shot forecasting without any training or fine-tuning.
        Uses the pre-trained Chronos model to generate median predictions.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
                Chronos expects univariate time series (n_features=1).
            y (torch.Tensor, optional): Target tensor (not used by Chronos).
                Defaults to None.
            x_mark (torch.Tensor, optional): Time features (not used by Chronos).
                Defaults to None.
            y_mark (torch.Tensor, optional): Target time features (not used by Chronos).
                Defaults to None.
            **kwargs: Additional arguments (not used).
        
        Returns:
            torch.Tensor: Point forecasts (median predictions) of shape
                (batch_size, forecast_horizon, n_features).
                
        Note:
            - Chronos expects input shape (batch, channels, length), so tensors
              are permuted internally
            - Returns median quantile predictions from the probabilistic forecast
            - For quantile predictions, use predict_prob() method
              
        Example:
            >>> input_tensor = torch.randn(32, 512, 1)  # batch=32, seq_len=512, features=1
            >>> predictions = model.predict(input_tensor)
            >>> # Returns: (32, 96, 1)  # forecast_horizon=96
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

    def _load(self, checkpoint_or_path: str):
        try:
            from chronos import Chronos2Pipeline
            self.pipeline = Chronos2Pipeline.from_pretrained(
                checkpoint_or_path,
                device_map='cpu',
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