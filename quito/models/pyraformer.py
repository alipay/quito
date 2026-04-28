# code adopted from https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/

from typing import Callable, Optional
import torch

from torch import nn
from math import ceil
from einops import rearrange, repeat

from quito.models.base import TimeSeriesModel
from quito.config.model import PyraFormerModelConfig
from quito.models.utils.pyraformer_utils import Model


class PyraFormer(TimeSeriesModel):
    """
    PyraFormer model wrapper for QuitoBench.

    PyraFormer uses pyramidal attention with multi-scale windows to capture
    both short-term and long-term dependencies in time series. The pyramidal
    structure enables efficient processing of long sequences.

    Key features:
    - Pyramidal attention with multi-scale windows
    - Efficient long-range dependency modeling
    - Bottleneck construction for efficiency
    - Optional RevIN normalization

    Reference:
        Liu et al. (2021). "Pyraformer: Low-Complexity Pyramidal Attention
        for Long-Range Time Series Modeling and Forecasting"
    """

    def __init__(self, config: PyraFormerModelConfig, local_rank: int = -1):
        """
        Initialize the PyraFormer model.

        Args:
            config (PyraFormerModelConfig): Model configuration containing
                architecture parameters (window_size, d_model, n_layer, etc.).
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
        """
        super().__init__(config, local_rank)
        config.device = self.device
        self.model = Model(config)
        self.use_revin = config.revin

    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):
        """
        Forward pass for time series forecasting.

        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor (not used by PyraFormer).
                Defaults to None.
            x_mark (torch.Tensor, optional): Time features (not used).
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features (not used).
                Defaults to None.
            **kwargs: Additional arguments (not used).

        Returns:
            torch.Tensor: Predicted time series of shape
                (batch_size, forecast_horizon, n_features).
        """
        predictions = self.model(x_enc=x, x_mark_enc=None, x_dec=None, x_mark_dec=None, pretrain=False)

        return predictions
