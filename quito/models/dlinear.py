import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from quito.models.base import TimeSeriesModel
from quito.config.model import DLinearModelConfig


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.
    
    Applies average pooling with padding to extract trend components
    from time series data.
    
    Args:
        kernel_size (int): Size of the moving average window.
        stride (int): Stride for the average pooling operation.
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        Apply moving average to input time series.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
        
        Returns:
            torch.Tensor: Trend component of shape (batch_size, seq_len, n_features).
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block.
    
    Decomposes time series into trend and seasonal (residual) components
    using moving average.
    
    Args:
        kernel_size (int): Kernel size for moving average decomposition.
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        Decompose time series into seasonal and trend components.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
        
        Returns:
            tuple: (seasonal, trend) where:
                - seasonal: Residual component (x - trend)
                - trend: Moving average component
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(TimeSeriesModel):
    """
    DLinear (Decomposition-Linear) model for QuitoBench.
    
    DLinear is a simple yet effective linear model that uses moving average
    decomposition to separate trend and seasonal components, then applies
    separate linear layers to each component.
    
    Key features:
    - Moving average decomposition (trend + seasonal)
    - Separate linear layers for trend and seasonal components
    - Support for individual (per-channel) or shared linear layers
    - Optional RevIN normalization
    - Strong baseline for time series forecasting
    
    Reference:
        Zeng et al. (2023). "Are Transformers Effective for Time Series Forecasting?"
    """
    def __init__(self,  config: DLinearModelConfig, local_rank: int = -1):
        """
        Initialize the DLinear model.
        
        Args:
            config (DLinearModelConfig): Model configuration containing
                architecture parameters (kernel_size, individual, revin, etc.).
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
        """
        super().__init__(config, local_rank)
        self.seq_len = config.seq_len
        self.pred_len = config.forecast_horizon
        self.revin = config.revin

        # Decompsition Kernel Size
        kernel_size = config.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = config.individual
        self.channels = config.enc_in

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs):
        """
        Forward pass for time series forecasting.
        
        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor (not used by DLinear).
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
        # x: [Batch, Input length, Channel]
        if self.revin:
            # Normalization from Non-stationary Transformer
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        if self.revin:
            x = x * stdev + means

        return x.permute(0,2,1) # to [Batch, Output length, Channel]