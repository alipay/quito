import torch.nn as nn
import torch

from quito.config.evaluation import MetricType


METRIC_MAPPING = {
    MetricType.MSE: nn.functional.mse_loss,
    MetricType.MAE: nn.functional.l1_loss
}


def cal_score(metric_name: MetricType, y_pred: torch.Tensor, y_true: torch.Tensor, **kwargs):
    metric_fn = get_metric_fn(metric_name)
    if not metric_fn:
        raise ValueError(f"Metric {metric_name} is not supported.")

    return metric_fn(y_pred, y_true, **kwargs)

def get_metric_fn(name: MetricType):
    return METRIC_MAPPING.get(name)