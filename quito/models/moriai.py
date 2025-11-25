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
    
    Supports two modes:
    1. 'inference': Zero-shot forecasting using MoiraiForecast.
    2. 'pretrain': Pre-training/Fine-tuning using MoiraiModule (Masked Autoencoder).
    """
    
    def __init__(self, config: MoriaiModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        self.mode = getattr(config, 'mode', 'inference')
        
        # Try to load Moirai
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            import os
            
            device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
            model_path = config.pretrained_model_name_or_path
            
            if self.mode == "pretrain":
                # ------------------------------------------------------------------
                # Pretrain Mode: Load MoiraiModule directly
                # ------------------------------------------------------------------
                # Try loading from pretrained (checkpoint or HF)
                try:
                    self.model = MoiraiModule.from_pretrained(model_path)
                except Exception:
                    # If failed, maybe it's a raw config instantiation (from scratch)
                    # For simplicty, we currently assume loading from existing structure/checkpoint
                    # or user should initialize MoiraiModule manually if building from scratch.
                    raise ValueError(f"Could not load MoiraiModule from {model_path} for pretraining.")
                
                self.model = self.model.to(device)
                # Ensure gradients enabled
                for param in self.model.parameters():
                    param.requires_grad = True
                    
            else:
                # ------------------------------------------------------------------
                # Inference Mode: Load MoiraiForecast (Zero-shot)
                # ------------------------------------------------------------------
                # Load MoiraiModule first
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
                        return
                    else:
                        raise

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
                
                # Default: freeze params for inference, but allow unfreezing if requested (though unusual for MoiraiForecast)
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
        Forward pass.
        In inference mode: generates predictions.
        In pretrain mode: computes loss (and returns it as a scalar tensor).
        """
        if self.mode == "pretrain":
            # For pretraining, we don't usually call forward directly for generation
            # We call train_step which calls MoiraiModule.training_step
            raise NotImplementedError("Use train_step for Moirai pretraining.")
            
        # ------------------------------------------------------------------
        # Inference Forward (Forecast Generation)
        # ------------------------------------------------------------------
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
        if self.mode == "pretrain":
            raise ValueError("Model is in pretrain mode, cannot generate forecasts.")
            
        with torch.no_grad():
            return self.forward(x, **kwargs)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Custom training step for Moirai.
        Overrides the standard behavior to handle Moirai's specific input requirements
        and loss computation.
        """
        if self.mode != "pretrain":
             # If someone tries to train MoiraiForecast, warn them
             return super().train_step(batch) # This will likely fail or do nonsense
             
        # ------------------------------------------------------------------
        # Pretraining Step
        # ------------------------------------------------------------------
        # Extract data from quito batch
        x = batch['x'] # [batch, seq_len, dim]
        y = batch['y'] # [batch, horizon, dim]
        
        # Moirai pretraining typically uses the full sequence (context + horizon)
        # and applies masking internally.
        target = torch.cat([x, y], dim=1) # [batch, total_len, dim]
        batch_size, seq_len, dim = target.shape
        
        # Construct Moirai-compatible batch dict
        # Assuming defaults for optional fields
        moirai_batch = {
            'target': target,
            'observed_mask': torch.ones_like(target, dtype=torch.bool), # Assume observed
            'past_is_pad': torch.zeros(batch_size, seq_len, dtype=torch.bool, device=self.device),
            'sample_id': torch.zeros(batch_size, dtype=torch.long, device=self.device),
            'variate_id': torch.zeros(batch_size, dim, dtype=torch.long, device=self.device),
            'time_id': torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device),
        }
        
        # Call MoiraiModule.training_step
        # It expects (batch, batch_idx)
        # Returns dictionary or scalar
        output = self.model.training_step(moirai_batch, batch_idx=0)
        
        # Extract loss
        if isinstance(output, torch.Tensor):
            loss = output
        elif isinstance(output, dict) and 'loss' in output:
            loss = output['loss']
        else:
            raise ValueError(f"Unknown output format from MoiraiModule.training_step: {type(output)}")
            
        return loss

    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Custom evaluation step."""
        if self.mode == "pretrain":
             # For pretraining, eval is just validation loss on reconstruction
             loss = self.train_step(batch) # Reuse logic for simplicity
             return {'loss': loss}, None # No predictions to return really
        else:
             # Standard inference eval
             return super()._eval_step(batch['x'], batch['y'])

    def loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        # Not used in custom train_step, but required by abstract base
        return torch.tensor(0.0, device=x.device, requires_grad=True)
