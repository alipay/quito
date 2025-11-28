from enum import Enum


class MetricType(Enum):
    MSE = 'mse'
    MAE = 'mae'
    CRPS = 'crps'
    RMSE = 'rmse'
    LOSS = 'loss'
