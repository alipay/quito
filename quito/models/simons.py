"""
Simons Model Wrapper - zero-shot inference with the Simons time series foundation models.
"""
import torch

from quito.models.base import TimeSeriesModel

MEDIAN = 4  # index of the 0.5 quantile in Simons's levels 0.1, ..., 0.9


class Simons(TimeSeriesModel):
    """
    Lightweight wrapper for the Simons time series foundation models (Abel AI Lab).

    Simons is a family of pre-trained models (25M, 60M, 178M and 297M parameters) for zero-shot
    probabilistic forecasting. The wrapper returns the median (0.5 quantile) as the point forecast.

    Installation:
        pip install "git+https://github.com/Abel-ai-lab/Simons"

    Configuration:
        model:
          model_name: 'Simons'
          checkpoint_path: 'abel-lab/Simons-297M'  # a Hugging Face id or a local folder

        Use `checkpoint_path`, not `pretrained_model_name_or_path`: the latter makes quito read
        the checkpoint's config.json through transformers' AutoConfig, which does not know the
        Simons format.

    Supports:
    - Zero-shot inference (predict)
    - Fine-tuning/Training (forward) - Not implemented

    Modes:
    - features M: the variates of a window form one multivariate group.
    - features S: each window is one univariate series.

    Reference:
        https://github.com/Abel-ai-lab/Simons
        https://huggingface.co/collections/abel-lab/simons-1
    """

    def __init__(self, config, local_rank: int = -1):
        """
        Initialize the Simons model.

        Args:
            config: Model configuration containing checkpoint_path and forecast_horizon.
            local_rank (int, optional): Local rank for distributed training. Defaults to -1.
        """
        super().__init__(config, local_rank)
        self.model = None
        self.use_revin = False  # Simons scales its inputs itself

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Training is not supported: Simons is evaluated zero-shot only."""
        raise NotImplementedError("Simons is evaluated zero-shot only.")

    def predict(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None,
                y_mark: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Generate point forecasts using Simons zero-shot inference.

        Args:
            x (torch.Tensor): Input time series of shape (batch_size, seq_len, n_features).
            y, x_mark, y_mark: Not used.

        Returns:
            torch.Tensor: Point forecasts (median) of shape
                (batch_size, forecast_horizon, n_features).
        """
        with torch.no_grad():
            history = x.permute(0, 2, 1).float()  # (N, C, L): the C variates of a window, one group
            quantiles = self.model.forecast(history, self.config.forecast_horizon)  # (N, C, H, 9)
            forecast = quantiles[..., MEDIAN]
            if not torch.isfinite(forecast).all():
                raise ValueError("non-finite Simons forecasts")
            return forecast.permute(0, 2, 1).to(x.device)  # (N, H, C)

    def _load(self, checkpoint_or_path):
        """Load a Simons checkpoint from a Hugging Face id or a local folder."""
        try:
            from simons import Simons as SimonsNet
        except ImportError as e:
            raise ImportError(
                "\n" + "=" * 80 + "\n"
                "Simons library not found!\n"
                'Install it with: pip install "git+https://github.com/Abel-ai-lab/Simons"\n'
                + "=" * 80
            ) from e
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SimonsNet.from_pretrained(checkpoint_or_path, device=device).eval()
