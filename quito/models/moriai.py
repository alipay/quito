"""
Moirai Model Wrapper - Simple and lightweight for time series forecasting.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from quito.models.base import TimeSeriesModel
from quito.config.model import MoriaiModelConfig


class MoriaiModel(TimeSeriesModel):
    """
    Lightweight wrapper for Salesforce Moirai time series foundation model.
    
    ⚠️ WARNING: Moirai is a pre-trained zero-shot inference model.
    It does NOT support gradient-based training or fine-tuning.
    Use this model for evaluation and inference only.
    
    For training time series models, use PatchTST, DLinear, or other trainable models.
    """
    
    def __init__(self, config: MoriaiModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        
        # Try to load Moirai
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            import os
            
            device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
            
            # Check if it's a local path or HuggingFace model ID
            model_path = config.pretrained_model_name_or_path
            
            # Load MoiraiModule from pretrained (works for both local and HF paths)
            try:
                moirai_module = MoiraiModule.from_pretrained(model_path)
            except Exception:
                # Fallback: if from_pretrained fails, try loading as checkpoint
                if os.path.exists(model_path):
                    self.model = MoiraiForecast.load_from_checkpoint(
                        checkpoint_path=model_path,
                        prediction_length=config.prediction_length,
                        context_length=config.context_length,
                        patch_size=config.patch_size,
                        num_samples=config.num_samples,
                    )
                    self.model = self.model.to(device)
                    # Enable gradients for training
                    for param in self.model.parameters():
                        param.requires_grad = True
                    return
                else:
                    raise
            
            # Create MoiraiForecast with the loaded module
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
            
            # Enable gradients for training (pretrained models have frozen params by default)
            for param in self.model.parameters():
                param.requires_grad = True
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
        """Forward pass - generates predictions with gradient support."""
        batch_size, seq_len, num_channels = x.shape
        
        # Prepare inputs for Moirai
        # past_target: historical values [batch, seq_len, target_dim]
        past_target = x
        
        # past_observed_target: mask for observed values [batch, seq_len, target_dim]
        # All ones means all values are observed
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool)
        
        # past_is_pad: padding mask [batch, seq_len]
        # All zeros means no padding
        past_is_pad = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Call Moirai's forward method with required arguments
        forecast = self.model(
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad,
        )
        
        # Debug: Check what we got
        import logging
        logging.info(f"Forecast type: {type(forecast)}, is tensor: {isinstance(forecast, torch.Tensor)}")
        if isinstance(forecast, torch.Tensor):
            logging.info(f"Forecast shape: {forecast.shape}, requires_grad: {forecast.requires_grad}, dtype: {forecast.dtype}")
        
        # Moirai typically returns samples [batch, num_samples, horizon, channels]
        # We need to compute mean across samples dimension and ensure correct horizon
        if isinstance(forecast, torch.Tensor):
            # If samples dimension exists, take mean
            if forecast.dim() == 4:  # [batch, num_samples, horizon, channels]
                forecast = forecast.mean(dim=1)  # [batch, horizon, channels]
            elif forecast.dim() == 3:  # Could be [batch, samples, horizon] or [batch, horizon, channels]
                # If middle dimension doesn't match forecast_horizon, it might be samples
                if forecast.shape[1] != self.forecast_horizon and forecast.shape[2] == num_channels:
                    # Assume [batch, samples, horizon] - average over samples
                    forecast = forecast.mean(dim=1)  # [batch, horizon]
                    forecast = forecast.unsqueeze(-1)  # [batch, horizon, 1]
            elif forecast.dim() == 2:  # [batch, horizon]
                forecast = forecast.unsqueeze(-1)  # [batch, horizon, channels]
            elif forecast.dim() == 0:  # Scalar - something went wrong
                raise ValueError(f"Moirai returned a scalar forecast. Shape: {forecast.shape}")
            
            # Final check: ensure horizon dimension matches
            if forecast.shape[1] != self.forecast_horizon:
                # Truncate or error
                if forecast.shape[1] > self.forecast_horizon:
                    logging.warning(f"Forecast has {forecast.shape[1]} steps, truncating to {self.forecast_horizon}")
                    forecast = forecast[:, :self.forecast_horizon, :]
                else:
                    raise ValueError(f"Forecast horizon mismatch: got {forecast.shape[1]}, expected {self.forecast_horizon}")
            
            # For univariate forecasting, use only the first channel
            if forecast.shape[-1] > num_channels:
                logging.info(f"Forecast has {forecast.shape[-1]} channels, taking first {num_channels} for univariate")
                forecast = forecast[:, :, :num_channels]
            elif forecast.shape[-1] < num_channels:
                # Expand if needed
                forecast = forecast.repeat(1, 1, num_channels)
            
            # Handle gradient issues: Moirai may not support gradients
            if not forecast.requires_grad and self.training:
                logging.warning("="*80)
                logging.warning("Moirai is in training mode but doesn't produce gradients!")
                logging.warning("This is expected - Moirai is designed for zero-shot inference only.")
                logging.warning("Use examples/eval_moriai.py for proper Moirai evaluation.")
                logging.warning("For training, use models like PatchTST instead.")
                logging.warning("="*80)
            
            logging.info(f"Final forecast shape: {forecast.shape}, requires_grad: {forecast.requires_grad}")
        else:
            raise ValueError(f"Expected tensor output, got {type(forecast)}")
        
        return forecast

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Generate forecasts using Moirai (inference mode, no gradients)."""
        with torch.no_grad():
            return self.forward(x, **kwargs)

    def loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss - for evaluation only."""
        y_pred = self.forward(x, **kwargs)
        return self.loss_fn(y_pred, y[:, -self.forecast_horizon:, :])

    def _eval_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Evaluation step."""
        y_pred = self.predict(x, **kwargs)
        
        score_dict = {}
        if self.metrics:
            from quito.metrics import cal_score
            for metric in self.metrics:
                score = cal_score(metric, y_pred=y_pred, y_true=y[:, -self.forecast_horizon:, :])
                score_dict[metric] = score
        
        return score_dict, y_pred
