# code adopted from https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/

from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from quito.models.base import TimeSeriesModel
from quito.config.model import ITransformerModelConfig
from quito.models.utils.itransformer_utils import (Encoder, 
                                                   EncoderLayer, 
                                                   FullAttention, 
                                                   AttentionLayer, 
                                                   DataEmbedding_inverted)



class Model(nn.Module):
    """
    iTransformer model architecture implementation.
    
    iTransformer inverts the standard transformer architecture by treating
    variates (features) as tokens and time points as features. This enables
    better cross-variate dependency learning for multivariate time series.
    
    Architecture:
    1. Inverted embedding: B L N -> B N E (variates as tokens)
    2. Encoder layers with self-attention
    3. Projection: B N E -> B N S -> B S N (forecast horizon)
    
    Reference:
        Liu et al. (2024). "iTransformer: Inverted Transformers Are Effective
        for Time Series Forecasting" https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        """
        Initialize iTransformer model.
        
        Args:
            configs: Configuration object containing:
                - seq_len: Input sequence length
                - forecast_horizon: Prediction length
                - d_model: Model dimension
                - n_heads: Number of attention heads
                - e_layers: Number of encoder layers
                - d_ff: Feed-forward dimension
                - dropout: Dropout rate
                - embed: Embedding type
                - freq: Frequency string
                - use_norm: Whether to use normalization
                - output_attention: Whether to output attention weights
                - class_strategy: Classification strategy
                - factor: Attention factor
                - activation: Activation function
        """
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.forecast_horizon
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,
                                                    configs.dropout)
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        self.projector = nn.Linear(configs.d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Generate forecasts from input time series.
        
        Args:
            x_enc (torch.Tensor): Encoder input of shape (B, L, N).
            x_mark_enc (torch.Tensor, optional): Time features for encoder.
            x_dec (torch.Tensor, optional): Decoder input (not used).
            x_mark_dec (torch.Tensor, optional): Time features for decoder (not used).
        
        Returns:
            tuple: (dec_out, attns) where:
                - dec_out: Forecasts of shape (B, S, N)
                - attns: Attention weights (if output_attention=True)
        """
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape # B L N
        # B: batch_size;    E: d_model; 
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # covariates (e.g timestamp) can be also embedded as tokens
        
        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N 
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N] # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attns


    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass of iTransformer.
        
        Args:
            x_enc (torch.Tensor): Encoder input of shape (B, L, N).
            x_mark_enc (torch.Tensor, optional): Time features for encoder.
            x_dec (torch.Tensor, optional): Decoder input (not used).
            x_mark_dec (torch.Tensor, optional): Time features for decoder (not used).
            mask (torch.Tensor, optional): Attention mask (not used).
        
        Returns:
            torch.Tensor or tuple: Forecasts of shape (B, pred_len, N).
                If output_attention=True, also returns attention weights.
        """
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class ITransformer(TimeSeriesModel):
    """
    iTransformer model wrapper for QuitoBench.
    
    iTransformer inverts the standard transformer by treating variates as tokens
    and time points as features. This architecture is particularly effective for
    multivariate time series forecasting as it enables better cross-variate
    dependency learning.
    
    Key features:
    - Inverted architecture: variates as tokens, time as features
    - Encoder-only architecture
    - Optional instance normalization
    - Cross-variate attention for multivariate forecasting
    
    Reference:
        Liu et al. (2024). "iTransformer: Inverted Transformers Are Effective
        for Time Series Forecasting"
    """
    def __init__(self, config: ITransformerModelConfig, local_rank: int = -1):
        """
        Initialize the iTransformer model.
        
        Args:
            config (ITransformerModelConfig): Model configuration containing
                architecture parameters (d_model, n_heads, e_layers, etc.).
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
        """
        super().__init__(config, local_rank)
        self.model = Model(config)
    
    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):
        """
        Forward pass for time series forecasting.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor (not used by iTransformer).
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
        predictions = self.model(x_enc=x, x_mark_enc=None, x_dec=None, x_mark_dec=None)

        return predictions
    