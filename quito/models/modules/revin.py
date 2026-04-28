# code from https://github.com/ts-kim/RevIN, with minor modifications

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) for time series.
    
    RevIN is a normalization technique that normalizes input time series
    and then denormalizes the output, enabling better generalization across
    different time series distributions. It's particularly effective for
    time series forecasting models.
    
    The normalization is reversible, meaning the original scale can be
    recovered after processing, which is crucial for forecasting tasks.
    
    Reference:
        Kim et al. (2022). "Reversible Instance Normalization for Accurate
        Time-Series Forecasting against Distribution Shift"
    
    Args:
        num_features (int): Number of features/channels in the time series.
        eps (float, optional): Small value added for numerical stability.
            Defaults to 1e-5.
        affine (bool, optional): If True, uses learnable affine parameters
            for normalization. Defaults to True.
        subtract_last (bool, optional): If True, subtracts the last value
            instead of mean. Defaults to False.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        Initialize RevIN module.
        
        Args:
            num_features (int): The number of features or channels.
            eps (float): A value added for numerical stability.
            affine (bool): If True, RevIN has learnable affine parameters.
            subtract_last (bool): If True, subtracts last value instead of mean.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        """
        Apply normalization or denormalization.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_features).
            mode (str): Operation mode. Must be 'norm' or 'denorm':
                - 'norm': Normalize the input (subtract mean/std)
                - 'denorm': Denormalize the output (restore original scale)
        
        Returns:
            torch.Tensor: Normalized or denormalized tensor of same shape as input.
            
        Raises:
            NotImplementedError: If mode is not 'norm' or 'denorm'.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        """
        Compute normalization statistics (mean/std or last value).
        
        Statistics are stored as instance variables for use in denormalization.
        
        Args:
            x (torch.Tensor): Input tensor for computing statistics.
        """
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        """
        Normalize input tensor.
        
        Args:
            x (torch.Tensor): Input tensor to normalize.
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        """
        Denormalize output tensor to restore original scale.
        
        Args:
            x (torch.Tensor): Normalized tensor to denormalize.
        
        Returns:
            torch.Tensor: Denormalized tensor with original scale restored.
        """
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x