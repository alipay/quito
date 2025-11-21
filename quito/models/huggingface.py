"""
Generic HuggingFace Model Wrapper - Simple and lightweight.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from quito.models.base import TimeSeriesModel
from quito.config.model import HuggingFaceModelConfig


class HuggingFaceModel(TimeSeriesModel):
    """
    Lightweight wrapper for any HuggingFace time series model.
    Supports GPU/CPU training and inference.
    """
    
    def __init__(self, config: HuggingFaceModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        
        from transformers import AutoModel
        
        device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
        self.model = AutoModel.from_pretrained(
            config.pretrained_model_name_or_path,
            trust_remote_code=config.trust_remote_code,
        )
        self.model = self.model.to(device)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Forward pass."""
        outputs = self.model(x, **kwargs)
        # Extract predictions from model output
        if hasattr(outputs, 'prediction_outputs'):
            return outputs.prediction_outputs
        elif hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state
        else:
            return outputs

    def loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss."""
        y_pred = self.forward(x, **kwargs)
        return self.loss_fn(y_pred, y[:, -self.forecast_horizon:, :])

    def _eval_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Evaluation step."""
        y_pred = self.forward(x, **kwargs)
        
        score_dict = {}
        if self.metrics:
            from quito.metrics import cal_score
            for metric in self.metrics:
                score = cal_score(metric, y_pred=y_pred, y_true=y[:, -self.forecast_horizon:, :])
                score_dict[metric] = score
        
        return score_dict, y_pred
