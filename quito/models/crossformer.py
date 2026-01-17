# code adopted from https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/

from typing import Callable, Optional
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch import Tensor
from math import ceil
from einops import rearrange, repeat

from quito.models.base import TimeSeriesModel
from quito.config.model import CrossFormerModelConfig
from quito.models.utils.crossformer_utils import (Encoder, 
                                                  Decoder,
                                                  FullAttention, 
                                                  AttentionLayer, 
                                                  TwoStageAttentionLayer,
                                                  DSW_embedding
                                                  )


class Model(nn.Module):
    """
    CrossFormer model architecture implementation.
    
    CrossFormer is a transformer-based model for time series forecasting that uses
    cross-dimension attention to capture dependencies across both time and feature
    dimensions. It employs segment-based processing and multi-scale encoding.
    
    The model consists of:
    - DSW (Dimension-Segment-Window) embedding for input representation
    - Multi-scale encoder with segment merging
    - Decoder with cross-attention to encoder outputs
    
    Args:
        data_dim (int): Number of input features/dimensions.
        in_len (int): Input sequence length.
        out_len (int): Output/prediction length.
        seg_len (int): Segment length for patch-based processing.
        win_size (int, optional): Window size for segment merging. Defaults to 4.
        factor (int, optional): Factor for cross-dimension attention. Defaults to 10.
        d_model (int, optional): Model dimension. Defaults to 512.
        d_ff (int, optional): Feed-forward dimension. Defaults to 1024.
        n_heads (int, optional): Number of attention heads. Defaults to 8.
        e_layers (int, optional): Number of encoder layers. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        baseline (bool, optional): Whether to use baseline (mean) prediction.
            Defaults to False.
        revin (bool, optional): Whether to use RevIN (Reversible Instance Normalization).
            Defaults to True.
        device (torch.device, optional): Device for computation. Defaults to 'cuda:0'.
    """
    def __init__(self, data_dim, in_len, out_len, seg_len, win_size = 4,
                factor=10, d_model=512, d_ff = 1024, n_heads=8, e_layers=3, 
                dropout=0.0, baseline=False, revin=True, device=torch.device('cuda:0')):
        super().__init__()
        self.data_dim = data_dim
        self.in_len = in_len
        self.out_len = out_len
        self.seg_len = seg_len
        self.merge_win = win_size
        self.revin = revin

        self.baseline = baseline

        self.device = device

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        self.pad_out_len = ceil(1.0 * out_len / seg_len) * seg_len
        self.in_len_add = self.pad_in_len - self.in_len

        # Embedding
        self.enc_value_embedding = DSW_embedding(seg_len, d_model)
        self.enc_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_in_len // seg_len), d_model))
        self.pre_norm = nn.LayerNorm(d_model)

        # Encoder
        self.encoder = Encoder(e_layers, win_size, d_model, n_heads, d_ff, block_depth = 1, \
                                    dropout = dropout,in_seg_num = (self.pad_in_len // seg_len), factor = factor)
        
        # Decoder
        self.dec_pos_embedding = nn.Parameter(torch.randn(1, data_dim, (self.pad_out_len // seg_len), d_model))
        self.decoder = Decoder(seg_len, e_layers + 1, d_model, n_heads, d_ff, dropout, \
                                    out_seg_num = (self.pad_out_len // seg_len), factor = factor)
        
    def forward(self, x_seq):
        """
        Forward pass of the CrossFormer model.
        
        Processes input time series through embedding, encoding, and decoding
        stages. Optionally applies RevIN normalization for better generalization.
        
        Args:
            x_seq (torch.Tensor): Input time series of shape (batch_size, in_len, data_dim).
        
        Returns:
            torch.Tensor: Predicted time series of shape (batch_size, out_len, data_dim).
        """
        if self.revin:
            # revin
            means = x_seq.mean(1, keepdim=True).detach() # N, 1, C
            x_seq = x_seq - means
            stdev = torch.sqrt(torch.var(x_seq, dim=1, keepdim=True, unbiased=False) + 1e-5) # N, 1, C
            x_seq /= stdev

        if (self.baseline):
            base = x_seq.mean(dim = 1, keepdim = True)
        else:
            base = 0
        batch_size = x_seq.shape[0]
        if (self.in_len_add != 0):
            x_seq = torch.cat((x_seq[:, :1, :].expand(-1, self.in_len_add, -1), x_seq), dim = 1)

        x_seq = self.enc_value_embedding(x_seq)
        x_seq += self.enc_pos_embedding
        x_seq = self.pre_norm(x_seq)
        
        enc_out = self.encoder(x_seq)

        dec_in = repeat(self.dec_pos_embedding, 'b ts_d l d -> (repeat b) ts_d l d', repeat = batch_size)
        predict_y = self.decoder(dec_in, enc_out)
        out = base + predict_y[:, :self.out_len, :]
        if self.revin:
            # revin
            out = out * stdev + means

        return out


class CrossFormer(TimeSeriesModel):
    """
    CrossFormer model wrapper for QuitoBench.
    
    CrossFormer is a transformer-based time series forecasting model that captures
    cross-dimension dependencies using two-stage attention (time and dimension stages).
    It processes time series in segments and uses multi-scale encoding for hierarchical
    pattern learning.
    
    Key features:
    - Cross-dimension attention for multivariate time series
    - Segment-based processing for efficiency
    - Multi-scale encoding with segment merging
    - RevIN normalization for better generalization
    
    Reference:
        Zhang & Yan (2022). "Crossformer: Transformer Utilizing Cross-Dimension
        Dependency for Multivariate Time Series Forecasting"
    """
    def __init__(self, config: CrossFormerModelConfig, local_rank: int = -1):
        """
        Initialize the CrossFormer model.
        
        Args:
            config (CrossFormerModelConfig): Model configuration containing
                architecture parameters (d_model, n_heads, e_layers, etc.).
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
        """
        super().__init__(config, local_rank)
        self.model = Model(data_dim=config.enc_in, 
                           in_len=config.seq_len, 
                           out_len=config.forecast_horizon, 
                           seg_len=config.seg_len, 
                           win_size=config.win_size,
                           factor=config.factor,
                           d_model=config.d_model,
                           d_ff=config.d_ff,
                           n_heads=config.n_heads,
                           e_layers=config.e_layers,
                           dropout=config.dropout,
                           baseline=config.baseline,
                           revin=config.revin,
                           device=self.device)
    
    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):
        """
        Forward pass for time series forecasting.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor (not used by CrossFormer).
                Defaults to None.
            x_mark (torch.Tensor, optional): Time features for input (not used).
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features for target (not used).
                Defaults to None.
            **kwargs: Additional arguments (not used).
        
        Returns:
            torch.Tensor: Predicted time series of shape
                (batch_size, forecast_horizon, n_features).
        """
        predictions = self.model(x_seq=x)

        return predictions
    