"""
Evaluation configuration classes for QuitoBench library.

This module defines evaluation metrics and related configuration classes
for assessing time series forecasting model performance.
"""

from enum import Enum


class MetricType(Enum):
    """
    Enumeration of supported evaluation metrics for time series forecasting.
    
    Defines all metrics available for model evaluation in QuitoBench:
    
    - MSE: Mean Squared Error
    - MAE: Mean Absolute Error
    - CRPS: Continuous Ranked Probability Score (for probabilistic forecasts)
    - RMSE: Root Mean Squared Error
    - MASE: Mean Absolute Scaled Error (uses in-sample naive baseline)
    - MASE_LEAK: MASE with leaky baseline (uses lagged future values)
    - MAPE: Mean Absolute Percentage Error
    - SMAPE: Symmetric Mean Absolute Percentage Error
    - SMASE: Smart MASE with automatic seasonality detection
    
    Usage:
        >>> metric = MetricType.MSE
        >>> score = cal_score(metric, predictions, targets)
    """
    MSE = 'mse'
    MAE = 'mae'
    CRPS = 'crps'
    RMSE = 'rmse'
    MASE = 'mase' # in-sample baseline version of MASE
    MASE_LEAK = 'mase_leak' # leakage version of MASE (use lagged future values as baseline)
    MAPE = 'mape'
    SMAPE = 'smape'
    SMASE = 'smase'