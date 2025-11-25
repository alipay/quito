"""
Moirai Model Wrapper - Simple and lightweight for time series forecasting.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, Any

from quito.models.base import TimeSeriesModel
from quito.config.model import MoriaiModelConfig


class MoriaiModel(TimeSeriesModel):
    """
    Lightweight wrapper for Salesforce Moirai time series foundation model.
    
    NOTE: This model is INFERENCE-ONLY. Pretraining is not supported.
    Use this model for zero-shot forecasting on your time series data.
    """
    
    def __init__(self, config: MoriaiModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        self.mode = getattr(config, 'mode', 'inference')
        
        # Warn if pretrain mode is requested
        if self.mode == "pretrain":
            import logging
            logging.warning(
                "Moirai pretraining is NOT supported. "
                "Model will be loaded in inference-only mode. "
                "See train_step() for details."
            )
            self.mode = "inference"  # Force inference mode
        
        # Try to load Moirai
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            import os
            
            device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
            model_path = config.pretrained_model_name_or_path
            
            # ------------------------------------------------------------------
            # Load MoiraiModule and wrap in MoiraiForecast (inference only)
            # ------------------------------------------------------------------
            try:
                moirai_module = MoiraiModule.from_pretrained(model_path)
            except Exception as e:
                raise ValueError(f"Could not load MoiraiModule from {model_path}: {e}")
            
            # Get dimensions from config or use defaults
            target_dim = getattr(config, 'target_dim', 1)
            feat_dynamic_real_dim = getattr(config, 'feat_dynamic_real_dim', 0)
            past_feat_dynamic_real_dim = getattr(config, 'past_feat_dynamic_real_dim', 0)
            
            self.model = MoiraiForecast(
                module=moirai_module,
                prediction_length=config.prediction_length,
                context_length=config.context_length,
                patch_size=config.patch_size,
                num_samples=config.num_samples,
                target_dim=target_dim,
                feat_dynamic_real_dim=feat_dynamic_real_dim,
                past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            )
            self.model = self.model.to(device)
            
            # Freeze parameters (inference only)
            for param in self.model.parameters():
                param.requires_grad = False 

        except ImportError as e:
            raise ImportError(
                "\n" + "="*80 + "\n"
                "Salesforce Moirai library (uni2ts) not found!\n"
                "\n"
                "To use the Moirai model, install it with:\n"
                "  pip install uni2ts\n"
                "\n"
                "Or install all optional model dependencies:\n"
                "  pip install -r requirements-optional.txt\n"
                "\n"
                "See requirements-optional.txt for more details on model-specific dependencies.\n"
                + "="*80
            ) from e

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for inference (forecast generation).
        
        Args:
            x: Input context tensor of shape (batch, seq_len, channels)
            y: Not used (for API compatibility)
            
        Returns:
            Forecast tensor of shape (batch, forecast_horizon, channels)
        """
        batch_size, seq_len, num_channels = x.shape
        
        # Prepare inputs for Moirai
        past_target = x
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        past_is_pad = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        forecast = self.model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        
        # Process output (samples -> mean)
        if isinstance(forecast, torch.Tensor):
            if forecast.dim() == 4:  # [batch, num_samples, horizon, channels]
                forecast = forecast.mean(dim=1)
            elif forecast.dim() == 3 and forecast.shape[1] != self.forecast_horizon:
                 # Assume [batch, samples, horizon]
                forecast = forecast.mean(dim=1).unsqueeze(-1)
            elif forecast.dim() == 2:
                forecast = forecast.unsqueeze(-1)
            
            # Truncate to horizon
            if forecast.shape[1] > self.forecast_horizon:
                forecast = forecast[:, :self.forecast_horizon, :]
            
            # Match channels
            if forecast.shape[-1] != num_channels:
                 if forecast.shape[-1] > num_channels:
                     forecast = forecast[:, :, :num_channels]
                 else:
                     forecast = forecast.repeat(1, 1, num_channels)

        return forecast

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate forecasts (inference only)."""
        with torch.no_grad():
            return self.forward(x, **kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Custom training step for Moirai.
        
        NOTE: Moirai pretraining is NOT supported in the quito framework due to:
        1. MoiraiForecast is designed for inference only (uses @torch.no_grad())
        2. MoiraiModule has fixed architecture from pretrained checkpoint that doesn't match custom configs
        3. Complex input format requirements (sample_id, variate_id, prediction_mask, etc.)
        
        For Moirai, use the model for zero-shot inference/evaluation only.
        """
        if self.mode == "pretrain":
            raise NotImplementedError(
                "\n" + "="*80 + "\n"
                "Moirai pretraining is NOT supported in the quito framework.\n"
                "\n"
                "Reasons:\n"
                "  1. MoiraiForecast is designed for inference only (uses @torch.no_grad())\n"
                "  2. MoiraiModule has fixed architecture from pretrained checkpoint\n"
                "  3. Complex input format requirements incompatible with quito's generic pipeline\n"
                "\n"
                "Recommendation:\n"
                "  - Use Moirai for zero-shot inference/evaluation (see examples/eval_moriai.py)\n"
                "  - For pretraining, use PatchTST or other trainable models\n"
                "  - To pretrain Moirai from scratch, use the official uni2ts library directly\n"
                + "="*80
            )
        
        # If someone tries to train in inference mode
        return super().train_step(batch)

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Custom evaluation step for inference."""
        return super()._eval_step(batch['x'], batch['y'])

    def loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # Not used in custom train_step, but required by abstract base
        return torch.tensor(0.0, device=x.device, requires_grad=True)
