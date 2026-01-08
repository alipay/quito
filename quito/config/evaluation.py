from enum import Enum


class MetricType(Enum):
    MSE = 'mse'
    MAE = 'mae'
    CRPS = 'crps'
    RMSE = 'rmse'
    MASE = 'mase' # in-sample baseline version of MASE
    MASE_LEAK = 'mase_leak' # leakage version of MASE (use lagged future values as baseline)
    MAPE = 'mape'
    SMAPE = 'smape'
    SMASE = 'smase'