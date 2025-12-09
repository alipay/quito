import torch
import torch.nn as nn
from einops import rearrange
import math
import numpy as np
from typing import Dict, List
import logging

from quito.models.utils.tstransformer_utils import masked_mean, masked_std
from quito.models.base import TimeSeriesModel
from quito.config.model import TSTransformerModelConfig
from quito.metrics import get_metric_fn, cal_score


class MultiHeadAttention(nn.Module):
    """Multi-head attention with GQA supported"""

    def __init__(self, d_model, n_heads, num_groups=None, d_k=None, d_v=None, attn_dropout=0.0, attn=False, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.num_groups = num_groups if num_groups is not None else n_heads  # used for GQA
        self.group_size = n_heads // self.num_groups
        self.n_heads = self.group_size * self.num_groups

        self.d_k = d_k if d_k is not None else d_model // n_heads
        self.d_v = d_v if d_v is not None else d_model // n_heads

        self.proj_q = nn.Linear(d_model, self.n_heads * self.d_k)
        self.proj_k = nn.Linear(d_model, self.num_groups * self.d_k)
        self.proj_v = nn.Linear(d_model, self.num_groups * self.d_v)
        self.out_v = nn.Linear(self.n_heads * self.d_v, d_model)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.attn = attn

    def forward(self, q, k, v, mask=None):
        b, l, d_model = q.shape
        _, s, _ = k.shape
        q = self.proj_q(q).view(b, l, self.n_heads, self.d_k)  # b, l, h, d
        k = self.proj_k(k).view(b, s, self.num_groups, self.d_k)  # b, s, g, d
        v = self.proj_v(v).view(b, s, self.num_groups, self.d_v)  # b, s, g, d_v
        k = k.unsqueeze(-2).expand(-1, -1, -1, self.group_size, -1)  # b, s, g, g_size, d
        v = v.unsqueeze(-2).expand(-1, -1, -1, self.group_size, -1)  # b, s, g, g_size, d_v
        k = k.reshape(b, s, self.num_groups * self.group_size, self.d_k)
        v = v.reshape(b, s, self.num_groups * self.group_size, self.d_v)
        # do attention calculation
        attn = torch.einsum('blhd, bshd -> bhls', q, k) / (self.d_k ** 0.5)  # (b, h, l, s)
        # mask and dropout
        if mask is not None:
            if mask.ndim == 2:
                # the shape of mask is (l, s)
                mask = mask.unsqueeze(0).unsqueeze(0)  # mask.shape = (1, 1, l, s)
            elif mask.ndim == 3:
                # the shape of mask is (b, l, s)
                mask = mask.unsqueeze(1)
            # the postion mask == 1 is being filled with -inf
            assert mask.ndim == attn.ndim
            mask = mask.to(q.device)
            attn.masked_fill_(mask.bool(), float('-inf'))

        # compute softmax
        attn = torch.softmax(attn, dim=-1)  # (b, h, l, s)
        # attn dropout
        attn = self.attn_dropout(attn)
        # compute the aggregation
        out = torch.einsum('bhls, bshd -> blhd', attn, v)  # (b, l, h, d_v)
        out = self.out_v(out.reshape(b, l, self.n_heads * self.d_v))

        if self.attn:
            return out, attn

        return out, None


class TransformerEncoderLayer(nn.Module):
    """transformer encoder layer, input x and perform MHA"""

    def __init__(self,
                 layer_type='time',
                 attn_type='full',
                 d_model=512,
                 d_ff=1024,
                 act=nn.GELU,
                 n_heads=8,
                 num_groups=None,
                 d_k=None,
                 d_v=None,
                 attn_dropout=0.0,
                 pre_norm=True,
                 norm_type='LayerNorm',
                 rope=True,
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            num_groups=num_groups,
            d_k=d_k,
            d_v=d_v,
            attn_dropout=attn_dropout,
            **kwargs
        )

        self.pre_norm = pre_norm
        if norm_type == 'LayerNorm':
            self.norm = nn.LayerNorm(d_model)  # affine applied at the last dim
        elif norm_type == 'RMSNorm':
            self.norm = nn.RMSNorm(d_model)  # affine applied at the last dim

        self.attn_type = attn_type
        self.layer_type = layer_type
        self.dropout = nn.Dropout(dropout)
        self.rope = rope
        self.act = act
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            self.act(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x, split_point=None, mask=None):
        b, l, d_model = x.shape
        layer_attn_mask = self.build_mask_from_layer(x, split_point)
        if mask:
            mask = mask * layer_attn_mask
        else:
            mask = layer_attn_mask
        # pre norm
        if self.pre_norm:
            x = self.norm(x)

        x, attn = self.attn(x, x, x, mask=mask)
        x = x + self.dropout(x)  # skip connection

        if not self.pre_norm:
            x = self.norm(x)

        if self.pre_norm:
            x = self.norm(x)

        x = x + self.dropout(self.ff(x))
        if not self.pre_norm:
            x = self.norm(x)
        return x, attn

    def build_mask_from_layer(self, x, split_point=None):
        b, l, d_model = x.shape
        if self.attn_type == 'full':
            mask = torch.zeros(b, l, l)  # no mask, full attention
        elif self.attn_type == 'causal':
            # build causal mask, where samples only attent to the previous tokens and itself (masked already)
            mask = torch.triu(torch.ones(b, l, l), diagonal=1)
        elif self.attn_type == 'cross':
            # cross attention between target tokens and context tokens
            assert split_point, "split_point is required for cross attention"
            mask_context = torch.zeros(b, l, split_point)  # context not masked
            mask_context_target = torch.ones(b, split_point, l - split_point)  # mask context -> target
            mask_target_target = torch.triu(torch.ones(b, l - split_point, l - split_point),
                                            diagonal=1)  # only attent to previous token
            mask_target = torch.cat([mask_context_target, mask_target_target], dim=1)
            mask = torch.cat([mask_context, mask_target], dim=-1)
        else:
            raise ValueError(f'attn type {self.attn_type} undefined !')

        assert mask.shape == (b, l, l)

        return mask.to(x.device)


class TransformerEncoder(nn.Module):
    """Transformer encoder block for multiple transformer layers"""

    def __init__(self,
                 layers,
                 d_model=512,
                 d_ff=1024,
                 act=nn.GELU,
                 n_heads=8,
                 num_groups=None,
                 d_k=None,
                 d_v=None,
                 attn_dropout=0.0,
                 pre_norm=True,
                 rope=True,
                 norm_type='LayerNorm',
                 dropout=0.0,
                 **kwargs):
        super().__init__()
        self.layers = layers
        module_lst = []
        for layer in layers:
            layer_type, attn_type = layer.split('_')
            former_layer = TransformerEncoderLayer(layer_type=layer_type,
                                                   attn_type=attn_type,
                                                   d_model=d_model,
                                                   d_ff=d_ff,
                                                   act=act,
                                                   n_heads=n_heads,
                                                   num_groups=num_groups,
                                                   d_k=d_k,
                                                   d_v=d_v,
                                                   attn_dropout=attn_dropout,
                                                   norm_type=norm_type,
                                                   rope=rope,
                                                   pre_norm=pre_norm,
                                                   dropout=dropout,
                                                   **kwargs
                                                   )
            module_lst.append(former_layer)

        self.module_lst = nn.ModuleList(module_lst)

    def forward(self, x, split_point=None, mask=None):
        B, L, C, D = x.shape
        for idx, layer in enumerate(self.module_lst):
            if layer.layer_type == 'time':
                x = rearrange(x, 'B L C D -> (B C) L D')  # group B and C to perform attention on time dimension
                x, attn = layer(x, split_point=split_point, mask=mask)  # B,
                x = rearrange(x, '(B C) L D -> B L C D', B=B, L=L, C=C, D=D)
            else:
                x = rearrange(x, 'B L C D -> (B L) C D')  # group B and L to perform attention on feature dimension
                x, attn = layer(x, split_point=split_point, mask=mask)  # B,
                x = rearrange(x, '(B L) C D -> B L C D', B=B, L=L, C=C, D=D)

            if attn is not None:
                torch.save(attn.cpu(), f'{idx}_{layer.layer_type}_{layer.attn_type}.pkl')

        return x


class SubspacePositionEncoder(nn.Module):
    def __init__(self, d_model, num_subspace=8):
        super().__init__()
        self.d_model = d_model
        self.num_subspace = num_subspace
        self.project = nn.Linear(d_model // num_subspace, d_model)
        self.seed = 16

    def forward(self, x, y):
        B, C, n_P, d_model = x.shape
        # B, C, n_P, d_model
        positional_embedding_rng = torch.Generator(device=x.device).manual_seed(
            self.seed
        )
        embs = torch.randn(
            (C, d_model // self.num_subspace),
            device=x.device,
            dtype=x.dtype,
            generator=positional_embedding_rng,
        )  # generate a random embedding for each feature
        pe = self.project(embs)[None, None]  # 1, 1, C, d_model
        x = x.permute(0, 2, 1, 3) + pe
        y = y.permute(0, 2, 1, 3) + pe

        return x.permute(0, 2, 1, 3), y.permute(0, 2, 1, 3)  # B, C, n_P, d_model


class LearnedPositionalEncoder(nn.Module):
    """
    Learned positional encoder for the input sequence
    """

    def __init__(self, num_embeddings, d_model, how='time'):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, d_model)
        self.how = how

    def forward(self, x, y):
        # B, C, n_P, d_model
        B, C, n_P, d_model = x.shape
        _, _, n_P_y, _ = y.shape
        if self.how == 'time':
            idx = torch.arange(1, n_P + n_P_y + 1, device=x.device)
        else:
            idx = torch.arange(1, C + 1, device=x.device)

        embeddings = self.embedding(idx)

        if self.how == 'time':
            embeddings = embeddings.unsqueeze(0).unsqueeze(1)
            x = x + embeddings[:, :, :n_P, :]
            y = y + embeddings[:, :, n_P:, :]
        else:
            embeddings = embeddings.unsqueeze(0).unsqueeze(2)
            x = x + embeddings
            y = y + embeddings

        return x, y


class IndexPostionEncoder(nn.Module):
    """
    Embed index relative positional information against the first token of target to the series,
    adopted from CHRONOS-v2
    """

    def __init__(self, max_context_len):
        super().__init__()
        self.max_context_len = max_context_len

    def forward(self, x, y):
        # B, C, L
        B, C, L_x = x.shape
        L_y = y.shape[2]
        x_idx = torch.flip(torch.arange(1, L_x + 1, device=x.device), dims=[0])  # [n_P_x, n_P_x-1, ..., 1]
        y_idx = torch.arange(0, L_y, device=x.device)  # [0, 1, ..., n_P_y-1]
        x_idx = x_idx / self.max_context_len
        y_idx = y_idx / self.max_context_len

        x_idx = x_idx.view(1, 1, L_x).expand(B, C, -1)
        y_idx = y_idx.view(1, 1, L_y).expand(B, C, -1)

        return x_idx, y_idx


class PatchPositionEmbedder(nn.Module):
    """
    Perform positional embedding for the input patched data
    """

    def __init__(self, patch_size, time_pe_type, feature_pe_type='subspace', d_model=512, max_context_len=6000,
                 max_features=200, **kwargs):
        super().__init__()
        if feature_pe_type == 'subspace':
            self.feature_pe = SubspacePositionEncoder(d_model=d_model, **kwargs)
        elif feature_pe_type == 'learned':
            self.feature_pe = LearnedPositionalEncoder(max_features, d_model, how='feature')
        else:
            self.feature_pe = None

        if time_pe_type == 'learned':
            self.time_pe = LearnedPositionalEncoder(max_context_len // patch_size, d_model, how='time')
        else:
            self.time_pe = None

    def forward(self, x, y):
        # input is patched, including masking
        B, C_x, n_P_x, D = x.shape  # B, num_patch, patch_size, C
        _, _, n_P_y, _ = y.shape
        if self.time_pe is not None:
            x, y = self.time_pe(x, y)

        if self.feature_pe is not None:
            x, y = self.feature_pe(x, y)

        return x, y


class FlattenHead(nn.Module):
    """
    For each forecast_horizon patch, project d_model -> patch_size -> quantiles
    """

    def __init__(self, d_model, patch_size, quantiles: int = 1):
        super().__init__()
        self._quantiles = None
        self.quantiles = quantiles
        self.patch_size = patch_size
        self.d_model = d_model
        self.decoder = nn.Linear(d_model, patch_size * quantiles)

    # def forward(self, x):
    #     B, n_P, C, D = x.shape
    #     x = self.decoder(x.permute(0, 2, 1, 3)) # B, C, n_P, P * q
    #     x = rearrange(x, 'b c n (p q) -> b c (n p) q', n=n_P, p=self.patch_size, q=self.quantiles)
    #     x = x.permute(0, 2, 1, 3) # B, forecast_horizon, C, q

    #     return x
    def forward(self, x):
        B, n_P, C, D = x.shape
        # first one as summarization
        # fetch first token,
        x = self.decoder(x.permute(0, 2, 1, 3))  # B, C, n_P - 1, P * q
        x = rearrange(x, 'b c n (p q) -> b c (n p) q', n=n_P, p=self.patch_size, q=self.quantiles)
        x = x.permute(0, 2, 1, 3)  # B, forecast_horizon, C, q

        return x

    # @property
    # def quantiles(self):
    #     return self._quantiles

    # @quantiles.setter
    # def quantiles(self, value):
    #     if isinstance(value, int):
    #         self._quantiles = np.linspace(0.01, 0.99, value).round(2)
    #     elif isinstance(value, list):
    #         self._quantiles = value
    #     elif isinstance(value, str):
    #         self._quantiles = eval(value)
    #     else:
    #         self._quantiles = None


class CEHead(nn.Module):
    pass


class Transformer(nn.Module):
    def __init__(self,
                 layers,
                 patch_size=8,
                 time_pe_type='learned',
                 feature_pe_type='subspace',
                 d_model=512,
                 d_ff=1024,
                 act='GELU',
                 n_heads=8,
                 num_groups=None,
                 d_k=None,
                 d_v=None,
                 attn_dropout=0.0,
                 pre_norm=True,
                 norm_type='LayerNorm',
                 rope=False,
                 dropout=0.0,
                 max_context_len=6000,
                 max_features=200,
                 revin=True,
                 **kwargs,
                 ):
        super().__init__()
        self.act = getattr(nn, act)
        self.encoder = TransformerEncoder(layers=layers,
                                          d_model=d_model,
                                          d_ff=d_ff,
                                          act=self.act,
                                          n_heads=n_heads,
                                          num_groups=num_groups,
                                          d_k=d_k,
                                          d_v=d_v,
                                          attn_dropout=attn_dropout,
                                          pre_norm=pre_norm,
                                          rope=rope,
                                          norm_type=norm_type,
                                          dropout=dropout)
        self.patch_size = patch_size
        self.max_context_len = max_context_len
        self.max_features = max_features

        # missing mask
        self.missing_embedding = nn.Embedding(1, d_model)
        # target mask
        self.target_embedding = nn.Embedding(1, d_model)
        # point projection
        self.point_embedding = nn.Linear(4, d_model)
        # patch projection
        self.patch_embedding = nn.Linear(self.patch_size * d_model, d_model)
        self.point_pe = IndexPostionEncoder(self.max_context_len)
        self.patch_pe = PatchPositionEmbedder(
            patch_size=patch_size,
            time_pe_type=time_pe_type,
            feature_pe_type=feature_pe_type,
            d_model=d_model,
            max_context_len=max_context_len,
            max_features=max_features
        )
        self.head = FlattenHead(d_model=d_model, patch_size=patch_size, quantiles=1)

    def forward(self, x, y, x_mark=None, y_mark=None):
        B, lookback_win, C = x.shape
        # We now only support MS, S, no covariates, the input y will be replaced by zeros
        _, forecast_horizon, num_targets = y.shape
        y = torch.zeros_like(y, device=y.device)
        # padding the lookback window and forecast horizon
        # pad lookback window from left and pad forecast horizon from right
        num_patches_x = math.ceil(lookback_win / self.patch_size)
        padding_size_x = num_patches_x * self.patch_size - lookback_win
        # append padding_x to the left of x
        padding_x = torch.zeros(B, padding_size_x, C).to(x.device)
        x = torch.cat([padding_x, x], dim=1)
        # calculate padding for y
        num_patches_y = math.ceil(forecast_horizon / self.patch_size)
        padding_size_y = num_patches_y * self.patch_size - forecast_horizon
        padding_y = torch.zeros(B, padding_size_y, num_targets).to(x.device)
        y = torch.cat([y, padding_y], dim=1)
        # create padding mask for x and y
        padding_mask_x = torch.zeros_like(x, device=x.device)
        padding_mask_y = torch.zeros_like(y, device=y.device)
        if padding_size_x > 0:
            padding_mask_x[:, :padding_size_x, :] = 1

        if padding_size_y > 0:
            padding_mask_y[:, -padding_size_y:, :] = 1

        # reset current length
        forecast_horizon_original = forecast_horizon
        lookback_win = x.shape[1]
        forecast_horizon = y.shape[1]
        # preprocess process:
        # first, replace target with zeros, great target mask, and fill na with missing embedding
        if num_targets < C:
            # need to construct a target with C channels
            missing_c = C - num_targets
            missing_targets = torch.zeros(B, forecast_horizon, missing_c).to(x.device)
            y = torch.cat([y, missing_targets], dim=-1)
            # generate missing mask for y
            missing_mask_y = torch.zeros(B, forecast_horizon, C).to(x.device)
            missing_mask_y[:, :, -missing_c:] = 1
            missing_mask_y[:, :, :missing_c] = padding_mask_y  # merge missing mask and padding mask
            target_mask_y = torch.zeros(B, forecast_horizon, C).to(x.device)
            target_mask_y[:, :, :num_targets] = 1
        else:
            target_mask_y = torch.ones(B, forecast_horizon, C).to(x.device)
            missing_mask_y = torch.zeros(B, forecast_horizon, C).to(x.device)
        # generate missing mask for x
        missing_mask_x = torch.isnan(x)
        # merge padding mask with missing mask, and convert all mask to bools
        missing_mask_x = (missing_mask_x.bool() | padding_mask_x.bool())
        missing_mask_y = missing_mask_y.bool()
        target_mask_y = target_mask_y.bool()
        # The embedding process:
        # first, fillna in x -> perform revin -> and project x, y point-wisely to d_model dimension,
        # second, fill na with missing embedding
        # third, perform patchify and project again P * d_model -> d_model
        # fourth, perform positional embedding
        x = torch.nan_to_num(x, nan=0.0)
        # here, perform instance-normalization
        x_mean = masked_mean(x, mask=~missing_mask_x, dim=1,
                             keepdim=True)  # here masked_mean expect mask to have 1=valid, our setting assume 1=masked
        x_std = masked_std(x, mask=~missing_mask_x, dim=1, keepdim=True) + 1e-8
        x = (x - x_mean) / x_std
        y = (y - x_mean) / x_std
        x = torch.asinh(x)  # arcsinh transform
        y = torch.asinh(y)  # arcsinh transform
        # add point-wise positional encoding
        x_idx, y_idx = self.point_pe(x.permute(0, 2, 1), y.permute(0, 2, 1))  # requires input to be B, C, L, ...
        x_idx = x_idx.permute(0, 2, 1).unsqueeze(-1)  # B, L, C, 1
        y_idx = y_idx.permute(0, 2, 1).unsqueeze(-1)  # B, L, C, 1
        target_x_mask = torch.zeros_like(x_idx).to(x.device)
        x = torch.cat([x.unsqueeze(-1), x_idx, missing_mask_x.unsqueeze(-1), target_x_mask], dim=-1)  # B, L, C, 4
        y = torch.cat([y.unsqueeze(-1), y_idx, missing_mask_y.unsqueeze(-1), target_mask_y.unsqueeze(-1)],
                      dim=-1)  # B, L, C, 4
        x = self.point_embedding(x)  # B, lb_win, C, d_model
        y = self.point_embedding(y)  # B, fh, C, d_model
        # get missing embedding
        # missing_embedding = self.missing_embedding.weight[[0]] # 1, d_model
        # missing_embedding_x = missing_embedding.view(1, 1, 1, -1).expand(B, lookback_win, C, -1)
        # missing_embedding_y = missing_embedding.expand(B, forecast_horizon, C, -1)
        # x = torch.where(missing_mask_x.unsqueeze(-1), missing_embedding_x, x) # B, lb_win, C, d_model
        # y = torch.where(missing_mask_y.unsqueeze(-1), missing_embedding_y, y) # B, fh, C, d_model
        # # add target embedding
        # target_embedding = self.target_embedding.weight[[0]]
        # target_embedding = target_embedding.view(1, 1, 1, -1).expand(B, forecast_horizon, C, -1)
        # zero_placeholder = torch.zeros_like(target_embedding).to(x.device)
        # target_embedding = torch.where(target_mask_y.unsqueeze(-1), target_embedding, zero_placeholder)
        # y = y + target_embedding
        # do patch seperately for x and y
        x = x.unfold(1, self.patch_size, self.patch_size)  # unfold at L dim, (B, n_P_x, C, d_model, P)
        y = y.unfold(1, self.patch_size, self.patch_size)  # unfold at L dim, (B, n_P_y, C, d_model, P)
        # patch embedding
        x = x.reshape(B, num_patches_x, C, -1)
        y = y.reshape(B, num_patches_y, C, -1)
        x = self.patch_embedding(x)  # (B, n_P_x, C, d_model)
        y = self.patch_embedding(y)  # (B, n_P_y, C, d_model)
        # add patch level positional embedding
        x, y = self.patch_pe(x.permute(0, 2, 1, 3), y.permute(0, 2, 1, 3))  # embedding requires B, C, n_P, D
        x = x.permute(0, 2, 1, 3)
        y = y.permute(0, 2, 1, 3)
        # final step: concat x y along the time dimension
        x = torch.cat([x, y], dim=1)
        # encode
        x = self.encoder(x, split_point=num_patches_x)  # B, n_P, C, d_model
        # decode with decoder head
        x = self.head(x[:, num_patches_x:, :, :])
        # only return the target of interest
        out = x[:, :forecast_horizon_original, :num_targets, :]
        x_mean = x_mean[:, :, :num_targets]
        x_std = x_std[:, :, :num_targets]

        return out, x_mean, x_std


class TSTransformer(TimeSeriesModel):
    def __init__(self, config: TSTransformerModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        self.model = Transformer(
            layers=config.layers,
            patch_size=config.patch_size,
            time_pe_type=config.time_pe_type,
            feature_pe_type=config.feature_pe_type,
            d_model=config.d_model,
            d_ff=config.d_ff,
            act=config.act,
            n_heads=config.n_heads,
            num_groups=config.num_groups,
            d_k=config.d_k,
            d_v=config.d_v,
            attn_dropout=config.attn_dropout,
            pre_norm=config.pre_norm,
            norm_type=config.norm_type,
            rope=config.rope,
            dropout=config.dropout,
            max_context_len=config.max_context_len,
            max_features=config.max_features,
            revin=config.revin,
        )

    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):
        return self.model.forward(x, y, x_mark, y_mark)

    def loss(self, x: torch.Tensor, y: torch.Tensor, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None,
             **kwargs) -> torch.Tensor:
        """
        Compute the loss given batch of inputs
        """
        # construct decoder input for encoder-decoder framework
        x, y_in, x_mark, y_mark = self._construct_model_input(x, y, x_mark, y_mark)
        y_pred, x_mean, x_std = self.forward(x=x, y=y_in, x_mark=x_mark, y_mark=y_mark, **kwargs)

        # use the normalized y to compute the loss
        y_pred = torch.sinh(y_pred) * x_std.unsqueeze(-1) + x_mean.unsqueeze(-1)

        y = y.unsqueeze(-1)

        return self.loss_fn(y_pred, y[:, -self.forecast_horizon:, :])

    def _eval_step(self, x: torch.Tensor, y: torch.Tensor, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None,
                   **kwargs) -> Dict[str, torch.Tensor]:
        """
        Perform a single evaluation step.

        Args:
            batch: Evaluation batch

        Returns:
            Dictionary containing loss and other metrics
        """
        # construct decoder input for encoder-decoder framework
        x, y_in, x_mark, y_mark = self._construct_model_input(x, y, x_mark, y_mark)
        y_pred, x_mean, x_std = self.forward(x=x, y=y_in, x_mark=x_mark, y_mark=y_mark, **kwargs)
        # use the normalized y to compute the loss
        # y = (y - x_mean) / x_std
        y_pred = torch.sinh(y_pred) * x_std.unsqueeze(-1) + x_mean.unsqueeze(-1)
        y = y.unsqueeze(-1)
        score_dict = {}
        for metric in self.metrics:
            score = cal_score(metric, y_pred=y_pred, y_true=y[:, -self.forecast_horizon:, :])
            score_dict[metric] = score

        return score_dict, y_pred

    def _construct_model_input(self, x, y, x_mark, y_mark):
        """ Construct decoder input using y for encoder-decoder framework """
        dec_in = torch.zeros_like(y[:, -self.forecast_horizon:, :])  # mask target

        return x, dec_in, x_mark, y_mark

    def predict(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        """
        Make predictions on input data.

        Args:
            x: Input tensor
            **kwargs: Additional prediction parameters

        Returns:
            Prediction tensor
        """
        self.eval()
        with torch.no_grad():
            y_pred, mean, std = self.forward(x=x, y=y, x_mark=x_mark, y_mark=y_mark, **kwargs)
            return torch.sinh(y_pred) * std + mean
