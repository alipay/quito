import numpy as np

from einops import rearrange

from quito.models.base import StatisticalModel
from quito.config.model import NaiveForecasterModelConfig


class NaiveForecaster(StatisticalModel):
    """
    Naive forecasting baseline methods.

    Implements various naive forecasting strategies:
    - Last value (naive)
    - Seasonal naive
    - Mean
    - Drift

    Example:
        >>> from quito.models import NaiveForecaster
        >>> config = NaiveForecasterModelConfig(method='last')
        >>> model = NaiveForecaster(config)
        >>> model.fit(train_data)
        >>> predictions = model.predict()
    """

    def __init__(self, config: NaiveForecasterModelConfig, local_rank=-1):
        """
        Initialize Naive Forecaster.

        Args:
            config: Model configuration
        """
        super().__init__(config, local_rank)

        self.method = getattr(config, 'method', 'last')  # 'last', 'seasonal', 'mean', 'drift'
        self.seasonal_period = getattr(config, 'seasonal_period', 12)
        self.forecast_horizon = config.forecast_horizon

    def _forward(self, x: np.ndarray, y: np.ndarray = None, x_mark: np.ndarray = None, y_mark: np.ndarray = None,
                 **kwargs) -> np.ndarray:
        """
        Fit naive forecaster (just stores relevant values).

        """
        N, L, C = x.shape
        train_data = rearrange(x, 'N L C -> (N C) L', N=N, L=L, C=C)  # N*C, L
        forecasts = self._fit_series(train_data, **kwargs)
        forecasts = rearrange(forecasts, '(N C) L -> N L C', N=N, L=self.forecast_horizon, C=C)

        return forecasts

    def _fit_series(self, train_data: np.ndarray, **kwargs):
        """
        Fit naive forecaster (just stores relevant values).
        """
        context = train_data
        if self.method == 'last':
            last_val = context[:, -1:]
            forecast = np.tile(last_val, (1, self.forecast_horizon))
        elif self.method == 'seasonal':
            seasonal = context[:, -self.seasonal_period:]
            n_repeats = int(np.ceil(self.forecast_horizon / self.seasonal_period))
            forecast = np.tile(seasonal, (1, n_repeats))[:, :self.forecast_horizon]
        elif self.method == 'mean':
            mean_val = context.mean(axis=1, keepdims=True)
            forecast = np.tile(mean_val, (1, self.forecast_horizon))
        else:
            raise ValueError(f"Unknown naive method: {self.method}")

        return forecast

    def _load(self, checkpoint_or_path: str):
        pass