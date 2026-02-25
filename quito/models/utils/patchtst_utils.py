import torch
from torch import nn
import math

__all__ = ['Transpose',
           'get_activation_fn',
           'moving_avg',
           'series_decomp',
           'PositionalEncoding',
           'SinCosPosEncoding',
           'Coord2dPosEncoding',
           'Coord1dPosEncoding',
           'positional_encoding',
           'series_decomp']


class Transpose(nn.Module):
    """
    Transpose layer for tensor dimension permutation.

    A simple wrapper around PyTorch's transpose operation that can optionally
    make the result contiguous in memory.

    Args:
        *dims: Dimension indices to transpose (e.g., 0, 1 for 2D transpose).
        contiguous (bool, optional): Whether to make output contiguous.
            Defaults to False.
    """

    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        """
        Transpose input tensor.

        Args:
            x (torch.Tensor): Input tensor to transpose.

        Returns:
            torch.Tensor: Transposed tensor, optionally contiguous.
        """
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


def get_activation_fn(activation):
    """
    Get activation function by name or return callable directly.

    Args:
        activation (str or callable): Activation function name ('relu', 'gelu')
            or a callable that returns an activation module.

    Returns:
        nn.Module: Activation function module.

    Raises:
        ValueError: If activation name is not recognized.

    Example:
        >>> relu = get_activation_fn('relu')
        >>> gelu = get_activation_fn('gelu')
    """
    if callable(activation):
        return activation()
    elif activation.lower() == "relu":
        return nn.ReLU()
    elif activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. You can use "relu", "gelu", or a callable')


# decomposition

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series.

    Applies average pooling with padding to extract trend components
    from time series data. Used for decomposition-based models.

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
    Series decomposition block for trend and seasonal separation.

    Decomposes time series into trend and seasonal (residual) components
    using moving average. Used in decomposition-based forecasting models.

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


# pos_encoding

def PositionalEncoding(q_len, d_model, normalize=True):
    """
    Generate sinusoidal positional encoding for transformer models.

    Creates positional encodings using sine and cosine functions of different
    frequencies, following the original Transformer paper.

    Args:
        q_len (int): Sequence length.
        d_model (int): Model dimension.
        normalize (bool, optional): Whether to normalize encodings.
            Defaults to True.

    Returns:
        torch.Tensor: Positional encoding of shape (q_len, d_model).
    """
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


SinCosPosEncoding = PositionalEncoding


def Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False):
    x = .5 if exponential else 1
    i = 0
    for i in range(100):
        cpe = 2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x) * (
                    torch.linspace(0, 1, d_model).reshape(1, -1) ** x) - 1
        pv(f'{i:4.0f}  {x:5.3f}  {cpe.mean():+6.3f}', verbose)
        if abs(cpe.mean()) <= eps:
            break
        elif cpe.mean() > eps:
            x += .001
        else:
            x -= .001
        i += 1
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (.5 if exponential else 1)) - 1)
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    """
    Create positional encoding based on specified type.

    Supports multiple positional encoding strategies:
    - None/zeros: Learnable or fixed zero encodings
    - normal/gauss: Gaussian initialization
    - uniform: Uniform initialization
    - lin1d/exp1d: 1D coordinate-based encodings
    - lin2d/exp2d: 2D coordinate-based encodings
    - sincos: Sinusoidal encodings

    Args:
        pe (str or None): Positional encoding type.
        learn_pe (bool): Whether to make encodings learnable.
        q_len (int): Sequence length.
        d_model (int): Model dimension.

    Returns:
        nn.Parameter: Positional encoding parameter.

    Raises:
        ValueError: If pe type is not recognized.
    """
    # Positional encoding
    if pe == None:
        W_pos = torch.empty((q_len, d_model))  # pe = None and learn_pe = False can be used to measure impact of pe
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == 'zero':
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'zeros':
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == 'normal' or pe == 'gauss':
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == 'uniform':
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    elif pe == 'lin1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == 'exp1d':
        W_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == 'lin2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == 'exp2d':
        W_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == 'sincos':
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', \
        'zeros', 'zero', uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)")
    return nn.Parameter(W_pos, requires_grad=learn_pe)