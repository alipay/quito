"""
Chronos Model Wrapper - Simple and lightweight for time series forecasting.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from quito.models.base import TimeSeriesModel
from quito.config.model import ChronosModelConfig


class ChronosModel(TimeSeriesModel):
    """
    Lightweight wrapper for Amazon Chronos time series foundation model.
    
    ⚠️ NOTE: Chronos is primarily designed for zero-shot inference.
    While fine-tuning is technically possible, it's not recommended.
    Use this model for evaluation and zero-shot forecasting only.
    
    For training time series models, use PatchTST, DLinear, or other trainable models.
    """
    
    def __init__(self, config: ChronosModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        
        # Try to load Chronos pipeline
        try:
            from chronos import ChronosPipeline
            device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
            self.pipeline = ChronosPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                device_map=device,
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
        """Forward pass - generates predictions."""
        # x: [batch, seq_len, channels]
        # For Chronos, we predict directly
        return self.predict(x, **kwargs)

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate forecasts using Chronos pipeline."""
        # Convert to format Chronos expects: [batch, seq_len]
        if x.dim() == 3:
            x = x.squeeze(-1)  # Remove channel dim if univariate
        
        # Generate forecasts
        with torch.no_grad():
            forecast = self.pipeline.predict(
                context=x,
                prediction_length=self.config.forecast_horizon,
                num_samples=self.config.num_samples,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            # forecast: [batch, num_samples, pred_len]
            # Return median prediction
            pred = forecast.median(dim=1).values  # [batch, pred_len]
            
        return pred.unsqueeze(-1)  # [batch, pred_len, 1]

    def loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss - for evaluation only (Chronos is pretrained)."""
        y_pred = self.forward(x, **kwargs)
        # Use the loss function set by trainer
        return self.loss_fn(y_pred, y[:, -self.forecast_horizon:, :])

    def _eval_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Evaluation step."""
        y_pred = self.predict(x, **kwargs)
        
        # Compute metrics
        score_dict = {}
        if self.metrics:
            from quito.metrics import cal_score
            for metric in self.metrics:
                score = cal_score(metric, y_pred=y_pred, y_true=y[:, -self.forecast_horizon:, :])
                score_dict[metric] = score
        
        return score_dict, y_pred
