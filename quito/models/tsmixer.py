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
    def __init__(self, config: TSMixerModelConfig, local_rank: int = -1):
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
    
    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):           # x: [Batch, Input length, Channel]
        predictions = self.model(x_hist=x)

        return predictions
    