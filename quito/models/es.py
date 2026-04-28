import logging
import numpy as np

from einops import rearrange

from quito.models.base import StatisticalModel
from quito.config.model import ESModelConfig


class ES(StatisticalModel):
    """
    Exponential Smoothing (ES) model for time series forecasting.
    """

    def __init__(self, config: ESModelConfig, local_rank=-1):
        """
        Initialize ETS model.

        Args:
            config: Model configuration with ETS parameters
        """
        super().__init__(config, local_rank)

        # ETS components
        self.trend = getattr(config, 'trend', 'add')  # 'add', 'mul', or None
        self.seasonal = getattr(config, 'seasonal', 'add')  # 'add', 'mul', or None
        self.seasonal_periods = getattr(config, 'seasonal_periods', None)
        self.damped_trend = getattr(config, 'damped_trend', False)
        self.forecast_horizon = config.forecast_horizon

    def _forward(self, x: np.ndarray, y: np.ndarray = None, x_mark: np.ndarray = None, y_mark: np.ndarray = None,
                 **kwargs) -> np.ndarray:
        """
        Fit ETS model on training data.

        Args:
            train_data: Training time series data
            **kwargs: Additional arguments


        Returns:
            Self
        """
        try:
            # use sktime interface
            from sktime.forecasting.exp_smoothing import ExponentialSmoothing
        except ImportError as e:
            raise e

        import warnings
        warnings.filterwarnings('ignore')

        N, L, C = x.shape
        train_data = rearrange(x, 'N L C -> N C L', N=N, L=L, C=C)
        # Create ETS model
        model = ExponentialSmoothing(
            trend=self.trend,
            seasonal=self.seasonal,
            sp=self.seasonal_periods,
            damped_trend=self.damped_trend
        )
        #
        preds = model.fit_predict(train_data, fh=list(np.arange(1, self.forecast_horizon + 1)))
        forecasts = rearrange(preds, 'N C L -> N L C', N=N, L=self.forecast_horizon, C=C)

        return forecasts

    def _load(self, checkpoint_or_path: str):
        pass