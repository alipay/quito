import torch.nn as nn
import torch
import numpy as np

from quito.config.evaluation import MetricType
from quito.utils.metrics import compute_naive_mae, detect_seasonality_fft


def cal_score(metric_name: MetricType, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
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
    return METRIC_MAPPING.get(name)

def MSE(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    loss = nn.functional.mse_loss(y_pred, y_true)
    return loss.item()

def MAE(y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    loss = nn.functional.l1_loss(y_pred, y_true)
    return loss.item()

def MASE(y_pred: torch.Tensor, y_true: torch.Tensor, x_train: torch.Tensor, **kwargs):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    x_train = x_train.numpy()
    return _mean_absolute_scaled_error(y_true, y_pred, x_train, **kwargs)

def MASE_LEAK(y_pred: torch.Tensor, y_true: torch.Tensor, x_train: torch.Tensor, **kwargs):
    y_pred = y_pred.numpy()
    y_true = y_true.numpy()
    x_train = x_train.numpy()

    return _mean_absolute_scaled_error(y_true, y_pred, x_train, leak=True, **kwargs)

def _mean_absolute_scaled_error(
    y_true, y_pred, x_train, sp=1, horizon_weight=None, multioutput="uniform_average", leak=False, **kwargs
):
    """Mean absolute scaled error (MASE).
    Adopted from 
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
    Calculate Mean Absolute Percentage Error (MAPE)
    
    Parameters:
    actual (array-like): Actual values
    forecast (array-like): Forecasted/predicted values
    
    Returns:
    float: MAPE value as a percentage
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
    Calculate Symmetric Mean Absolute Percentage Error (sMAPE)
    
    Parameters:
    actual (array-like): Actual values
    forecast (array-like): Forecasted/predicted values
    
    Returns:
    float: sMAPE value as a percentage
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
    Calculate Root Mean Square Error (RMSE)
    
    Parameters:
    y_true (torch.Tensor): Actual values
    y_pred (torch.Tensor): Predicted values
    
    Returns:
    float: RMSE value
    """
    actual = y_true.numpy()
    forecast = y_pred.numpy()
    
    rmse_value = np.sqrt(np.mean((actual - forecast) ** 2))
    return rmse_value


def SMASE(y_pred, y_true, x_train, eps=1e-6, min_period=2, peak_threshold=0.1, **kwargs):
    """
    Smart SMASE: auto-detects seasonality per series & channel using FFT.
    
    Shapes:
        y_train: (N, L_t, C)
        y_true:  (N, L,  C)
        y_pred:  (N, L,  C)
    
    Returns:
        sMASE: scalar (mean over N, L, C)
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
