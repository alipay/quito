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
    def __init__(self, config: PyraFormerModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        config.device = self.device
        self.model = Model(config)
    
    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):           # x: [Batch, Input length, Channel]
        predictions = self.model(x_enc=x, x_mark_enc=None, x_dec=None, x_mark_dec=None, pretrain=False)

        return predictions
    