# code adopted from https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from quito.models.base import TimeSeriesModel
from quito.config.model import TSMixerModelConfig
from quito.models.utils.tsmixer_utils import TSMixer as TSMixerBase


class TSMixer(TimeSeriesModel):
    """
    TSMixer model wrapper for QuitoBench.

    TSMixer is a lightweight MLP-based model that uses mixing layers to capture
    both temporal and cross-feature dependencies. It's an efficient alternative
    to transformer models for time series forecasting.

    Key features:
    - MLP-based architecture (no attention mechanism)
    - Mixing layers for temporal and feature mixing
    - Lightweight and efficient
    - Optional RevIN normalization

    Reference:
        Ekambaram et al. (2023). "TSMixer: Lightweight MLP-Mixer Model for
        Multivariate Time Series Forecasting"
    """

    def __init__(self, config: TSMixerModelConfig, local_rank: int = -1):
        """
        Initialize the TSMixer model.

        Args:
            config (TSMixerModelConfig): Model configuration containing
                architecture parameters (num_blocks, d_ff, norm_type, etc.).
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
        """
        super().__init__(config, local_rank)
        self.model = TSMixerBase(
            sequence_length=config.seq_len,
            prediction_length=config.forecast_horizon,
            input_channels=config.enc_in,
            output_channels=config.c_out,
            activation_fn=config.activation,
            num_blocks=config.num_blocks,
            dropout_rate=config.dropout,
            ff_dim=config.d_ff,
            normalize_before=config.pre_norm,
            norm_type=config.norm_type,
            revin=config.revin
        )
        self.use_revin = config.revin

    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):
        """
        Forward pass for time series forecasting.

        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor (not used by TSMixer).
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
        predictions = self.model(x_hist=x)

        return predictions
