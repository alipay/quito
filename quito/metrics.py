import torch.nn as nn
import torch
import numpy as np

from quito.config.evaluation import MetricType
from quito.utils.metrics import compute_naive_mae, detect_seasonality_fft


def cal_score(metric_name: MetricType, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    """
    Calculate evaluation metric score for predictions.
    
    Computes a specified evaluation metric by looking up the metric function
    and applying it to predictions and ground truth. Automatically moves
    tensors to CPU for consistency across different model architectures.
    
    Args:
        metric_name (MetricType): Name of the metric to compute (e.g., MSE, MAE, MASE).
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        **kwargs: Additional arguments passed to the metric function.
            Common arguments include:
            - x_train: Training data (required for MASE, SMASE)
            - y_pred_quantile: Quantile predictions (for probabilistic metrics)
    
    Returns:
        float: Computed metric score as a Python float.
        
    Raises:
        ValueError: If the metric name is not supported or metric function not found.
        
    Example:
        >>> score = cal_score(MetricType.MSE, predictions, targets)
        >>> mase_score = cal_score(MetricType.MASE, pred, true, x_train=train_data)
    """
    metric_fn = get_metric_fn(metric_name)
    if not metric_fn:
        raise ValueError(f"Metric {metric_name} is not supported.")

    # move y_pred, y_true to cpu to have consistency between different model architectures
    y_pred = y_pred.cpu()
    y_true = y_true.cpu()
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            kwargs[k] = v.cpu()

    score = metric_fn(y_pred=y_pred, y_true=y_true, **kwargs)
    if isinstance(score, torch.Tensor) or isinstance(score, np.ndarray):
        score = score.item()
    
    return score

def get_metric_fn(name: MetricType):
    """
    Get the metric function for a given metric name.
    
    Args:
        name (MetricType): Metric type enum value.
    
    Returns:
        Callable: Metric function that computes the score, or None if not found.
        
    Example:
        >>> mse_fn = get_metric_fn(MetricType.MSE)
        >>> score = mse_fn(y_pred, y_true)
    """
    return METRIC_MAPPING.get(name)

def MSE(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    """
    Calculate Mean Squared Error (MSE).
    
    Computes the average squared difference between predicted and actual values.
    Lower values indicate better predictions.
    
    Args:
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        **kwargs: Additional arguments (unused for MSE).
    
    Returns:
        float: MSE value (non-negative).
        
    Example:
        >>> mse = MSE(predictions, targets)
        >>> # Returns: 0.25 (for example)
    """
    loss = nn.functional.mse_loss(y_pred, y_true)
    return loss.item()

def MAE(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    """
    Calculate Mean Absolute Error (MAE).
    
    Computes the average absolute difference between predicted and actual values.
    Less sensitive to outliers than MSE.
    
    Args:
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        **kwargs: Additional arguments (unused for MAE).
    
    Returns:
        float: MAE value (non-negative).
        
    Example:
        >>> mae = MAE(predictions, targets)
        >>> # Returns: 0.5 (for example)
    """
    loss = nn.functional.l1_loss(y_pred, y_true)
    return loss.item()

def MASE(y_pred: torch.Tensor, y_true: torch.Tensor, x_train: torch.Tensor, **kwargs):
    """
    Calculate Mean Absolute Scaled Error (MASE).
    
    MASE scales the MAE by the MAE of a naive seasonal forecast on training data,
    making it scale-independent and comparable across different time series.
    Values < 1 indicate better than naive forecast.
    
    Args:
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        x_train (torch.Tensor): Training data used for naive baseline computation.
        **kwargs: Additional arguments passed to _mean_absolute_scaled_error:
            - sp: Seasonal period (default: 1 for non-seasonal)
            - horizon_weight: Optional weights for different forecast horizons
    
    Returns:
        float: MASE value (non-negative, typically < 1 for good models).
        
    Reference:
        Hyndman & Koehler (2006). "Another look at measures of forecast accuracy."
    """
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    x_train = x_train.numpy()
    return _mean_absolute_scaled_error(y_true, y_pred, x_train, **kwargs)

def MASE_LEAK(y_pred: torch.Tensor, y_true: torch.Tensor, x_train: torch.Tensor, **kwargs):
    """
    Calculate Mean Absolute Scaled Error with Leaky Baseline (MASE-LEAK).
    
    Similar to MASE but uses a "leaky" baseline that has access to future
    information (uses y_true[:, :-1] as naive forecast for y_true[:, 1:]).
    This provides a more lenient baseline for evaluation.
    
    Args:
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        x_train (torch.Tensor): Training data (used for compatibility, not in leaky baseline).
        **kwargs: Additional arguments (unused).
    
    Returns:
        float: MASE-LEAK value (typically lower than standard MASE).
        
    Note:
        This metric is useful for understanding model performance relative to
        an optimistic baseline, but should be interpreted with caution.
    """
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    x_train = x_train.numpy()

    return _mean_absolute_scaled_error(y_true, y_pred, x_train, leak=True, **kwargs)

def _mean_absolute_scaled_error(
    y_true, y_pred, x_train, sp=1, horizon_weight=None, multioutput="uniform_average", leak=False, **kwargs
):
    """
    Compute Mean Absolute Scaled Error (MASE) implementation.
    
    Internal function that implements the MASE calculation. Scales the prediction
    error by the error of a naive seasonal forecast baseline.
    
    Args:
        y_true: Ground truth values (numpy array).
        y_pred: Predicted values (numpy array).
        x_train: Training data for computing naive baseline (numpy array).
        sp (int, optional): Seasonal period for naive forecast. Defaults to 1.
        horizon_weight: Optional weights for different forecast horizons (unused).
        multioutput (str, optional): Aggregation method (unused, kept for compatibility).
            Defaults to "uniform_average".
        leak (bool, optional): If True, use leaky baseline (access to future info).
            Defaults to False.
        **kwargs: Additional arguments (unused).
    
    Returns:
        float: MASE value.
        
    Reference:
        Implementation adopted from sktime:
        https://github.com/sktime/sktime/blob/v0.40.1/sktime/performance_metrics/forecasting/_functions.py#L342-L466
    """
    if leak:
        # use leaky baseline
        y_pred_naive = y_true[:, :-1, :]
        mae_naive = np.mean(np.abs(y_true[:, 1:, :] - y_pred_naive))
    else:
        # naive seasonal prediction
        x_train = np.asarray(x_train)
        y_pred_naive = x_train[:, :-sp, :]
        # mean absolute error of naive seasonal prediction
        mae_naive = np.mean(np.abs(x_train[:, sp:, :] - y_pred_naive))

    mae_pred = np.mean(np.abs(y_true - y_pred)) 
    loss = mae_pred / np.maximum(mae_naive, 1e-6)

    return loss.item()

def MAPE(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Computes the average absolute percentage error between predictions and
    ground truth. Returns NaN if all actual values are zero.
    
    Args:
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        **kwargs: Additional arguments (unused for MAPE).
    
    Returns:
        float: MAPE value as a percentage (0-100), or NaN if all actuals are zero.
        
    Note:
        MAPE can be problematic when actual values are close to zero.
        Consider using SMAPE for symmetric percentage error.
        
    Example:
        >>> mape = MAPE(predictions, targets)
        >>> # Returns: 5.2 (for 5.2% error)
    """
    actual = y_true.numpy()
    forecast = y_pred.numpy()
    
    # Handle division by zero - mask where actual is 0
    mask = actual != 0
    if not np.any(mask):
        return np.nan  # or float('inf')
    
    mape_value = np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100
    return mape_value.item()

def SMAPE(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    Computes the symmetric version of MAPE, which treats over- and under-predictions
    symmetrically. More robust than MAPE when actual values are close to zero.
    
    Args:
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        **kwargs: Additional arguments (unused for SMAPE).
    
    Returns:
        float: SMAPE value as a percentage (0-200), or 0.0 if both actual and
            predicted are zero.
        
    Note:
        SMAPE ranges from 0 to 200%, where 0% is perfect prediction.
        Lower values indicate better predictions.
        
    Example:
        >>> smape = SMAPE(predictions, targets)
        >>> # Returns: 3.5 (for 3.5% symmetric error)
    """
    actual = y_true.numpy()
    forecast = y_pred.numpy()
    
    # Handle case where both actual and forecast are 0
    denominator = np.abs(actual) + np.abs(forecast)
    mask = denominator != 0
    
    if not np.any(mask):
        return 0.0
    
    smape_value = np.mean(
        np.abs(actual[mask] - forecast[mask]) / denominator[mask]
    ) * 200  # Multiply by 200 for percentage (2 * 100)
    
    return smape_value.item()

def RMSE(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    """
    Calculate Root Mean Square Error (RMSE).
    
    Computes the square root of the mean squared error, providing error in the
    same units as the target variable. More sensitive to outliers than MAE.
    
    Args:
        y_pred (torch.Tensor): Predicted values tensor.
        y_true (torch.Tensor): Ground truth values tensor.
        **kwargs: Additional arguments (unused for RMSE).
    
    Returns:
        float: RMSE value (non-negative, same units as target).
        
    Note:
        RMSE = sqrt(MSE), so it's always >= MAE for the same predictions.
        
    Example:
        >>> rmse = RMSE(predictions, targets)
        >>> # Returns: 0.5 (for example, in same units as target)
    """
    actual = y_true.numpy()
    forecast = y_pred.numpy()
    
    rmse_value = np.sqrt(np.mean((actual - forecast) ** 2))
    return rmse_value


def SMASE(y_pred, y_true, x_train, eps=1e-6, min_period=2, peak_threshold=0.1, **kwargs):
    """
    Smart Mean Absolute Scaled Error (SMASE) with automatic seasonality detection.
    
    Computes MASE with automatic detection of seasonal period per time series
    and channel using FFT analysis. This makes the metric adaptive to different
    seasonal patterns in multivariate time series.
    
    Args:
        y_pred (torch.Tensor or np.ndarray): Predicted values of shape (N, L, C),
            where N is batch size, L is forecast horizon, C is number of channels.
        y_true (torch.Tensor or np.ndarray): Ground truth values of shape (N, L, C).
        x_train (torch.Tensor or np.ndarray): Training data of shape (N, L_t, C),
            where L_t is training sequence length.
        eps (float, optional): Small epsilon to prevent division by zero.
            Defaults to 1e-6.
        min_period (int, optional): Minimum seasonal period to consider.
            Defaults to 2.
        peak_threshold (float, optional): Threshold for FFT peak detection.
            Defaults to 0.1.
        **kwargs: Additional arguments (unused).
    
    Returns:
        float: SMASE value averaged over all samples and channels.
        
    Note:
        This metric automatically adapts to each time series' seasonal pattern,
        making it more robust for heterogeneous datasets.
        
    Example:
        >>> smase = SMASE(predictions, targets, train_data)
        >>> # Automatically detects seasonality per series/channel
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    x_train = np.asarray(x_train)
    
    N, L, C = y_true.shape
    total_mase = 0.0
    count = 0

    for n in range(N):
        for c in range(C):
            # 1. Compute model MAE for this (n,c)
            mae_model = np.mean(np.abs(y_true[n, :, c] - y_pred[n, :, c]))
            
            # 2. Detect seasonality from training data
            train_series = x_train[n, :, c]
            m = detect_seasonality_fft(
                train_series,
                min_period=min_period,
                max_period=len(train_series) // 2,
                peak_threshold=peak_threshold
            )
            
            # 3. Compute naive MAE on training
            mae_naive = compute_naive_mae(train_series, m=m)
            
            # 4. Safe MASE
            denom = max(eps, mae_naive)
            mase_nc = mae_model / denom
            
            total_mase += mase_nc
            count += 1

    return (total_mase / count).item()


METRIC_MAPPING = {
    MetricType.MSE: MSE,
    MetricType.MAE: MAE,
    MetricType.MASE: MASE,
    MetricType.MASE_LEAK: MASE_LEAK,
    MetricType.MAPE: MAPE,
    MetricType.SMAPE: SMAPE,
    MetricType.RMSE: RMSE,
    MetricType.SMASE: SMASE,
}
