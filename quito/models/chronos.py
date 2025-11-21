"""
Chronos Model Wrapper - Simple and lightweight for time series forecasting.
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional, Union

from quito.models.base import TimeSeriesModel
from quito.config.model import ChronosModelConfig


class ChronosModel(TimeSeriesModel):
    """
    Lightweight wrapper for Amazon Chronos time series foundation model.
    
    Supports:
    - Zero-shot inference (predict)
    - Fine-tuning/Training (forward)
    - Probabilistic forecasting (predict_prob)
    """
    
    def __init__(self, config: ChronosModelConfig, local_rank: int = -1):
        super().__init__(config, local_rank)
        
        # Try to load Chronos pipeline
        try:
            from chronos import ChronosPipeline
            device = f"cuda:{local_rank}" if local_rank >= 0 and torch.cuda.is_available() else "cpu"
            self.pipeline = ChronosPipeline.from_pretrained(
                config.pretrained_model_name_or_path,
                device_map=device,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            self.model = self.pipeline.model
            self.tokenizer = self.pipeline.tokenizer
        except ImportError as e:
            raise ImportError(
                "\n" + "="*80 + "\n"
                "Amazon Chronos library not found!\n"
                "\n"
                "To use the Chronos model, install it with:\n"
                "  pip install git+https://github.com/amazon-science/chronos-forecasting.git\n"
                "\n"
                "Or install all optional model dependencies:\n"
                "  pip install -r requirements-optional.txt\n"
                "\n"
                "See requirements-optional.txt for more details on model-specific dependencies.\n"
                + "="*80
            ) from e

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: Context time series [batch, seq_len, 1]
            y: Target time series [batch, pred_len, 1]
            
        Returns:
            Output tensor (loss or logits)
        """
        if y is None:
            # If no target provided, assume inference/generation
            return self.predict(x, **kwargs)
             
        # Prepare data for training (fine-tuning)
        if x.dim() == 3:
            x = x.squeeze(-1)
        if y.dim() == 3:
            y = y.squeeze(-1)
            
        # Tokenize context (x) and target (y)
        # We use the tokenizer to convert float time series to token IDs
        # This relies on the Chronos tokenizer API.
        try:
            context_ids, context_mask, _ = self.tokenizer.context_input_transform(x)
            target_ids, target_mask, _ = self.tokenizer.context_input_transform(y)
        except AttributeError:
            # Fallback or error if API differs
            raise NotImplementedError(
                "Could not access 'context_input_transform' on Chronos tokenizer. "
                "Please ensure you have the latest version of the chronos-forecasting library."
            )
        
        # Forward pass through T5 model
        # T5 expects: input_ids, attention_mask, labels
        # Labels are automatically shifted right for decoder training in HF T5
        outputs = self.model(
            input_ids=context_ids.to(self.device),
            attention_mask=context_mask.to(self.device),
            labels=target_ids.to(self.device)
        )
        
        return outputs.loss

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Generate point forecasts (median).
        """
        if x.dim() == 3:
            x = x.squeeze(-1)
            
        with torch.no_grad():
            forecast = self.pipeline.predict(
                x,
                prediction_length=self.config.forecast_horizon,
                num_samples=self.config.num_samples,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            # forecast is likely a generator or custom object in some versions, 
            # but in others a tensor.
            
            if hasattr(forecast, 'median'):
                pred = forecast.median(dim=1).values
            elif isinstance(forecast, torch.Tensor):
                pred = forecast.median(dim=1).values
            else:
                # Fallback for tensor
                pred = torch.tensor(forecast).median(dim=1).values
            
        return pred.unsqueeze(-1)

    def predict_prob(self, x: torch.Tensor, quantiles: List[float] = [0.1, 0.5, 0.9], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate probabilistic forecasts.
        
        Args:
            x: Input tensor [batch, seq_len, 1]
            quantiles: List of quantiles to compute (e.g. [0.1, 0.5, 0.9])
            
        Returns:
            Dictionary with:
                - 'samples': [batch, num_samples, pred_len, 1]
                - 'quantiles': [batch, len(quantiles), pred_len, 1]
                - 'mean': [batch, pred_len, 1]
        """
        if x.dim() == 3:
            x = x.squeeze(-1)
            
        with torch.no_grad():
            # Get raw samples from pipeline
            forecast = self.pipeline.predict(
                x,
                prediction_length=self.config.forecast_horizon,
                num_samples=self.config.num_samples,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
                top_p=self.config.top_p,
            )
            
            # Handle forecast object types
            if hasattr(forecast, 'samples'):
                # If it's a Forecast object (GluonTS style)
                samples_tensor = torch.tensor(forecast.samples) # [batch, num_samples, pred_len]
            elif isinstance(forecast, torch.Tensor):
                samples_tensor = forecast
            else:
                samples_tensor = torch.tensor(forecast)

            # Ensure samples are on correct device
            samples_tensor = samples_tensor.to(self.device)
            
            # Calculate stats
            # Quantiles: [len(quantiles), batch, pred_len] -> permute to [batch, len(quantiles), pred_len]
            q_tensor = torch.tensor(quantiles, device=self.device)
            q_values = torch.quantile(samples_tensor, q_tensor, dim=1).permute(1, 0, 2)
            
            mean_pred = samples_tensor.mean(dim=1)
            
            # Add channel dimension
            samples_out = samples_tensor.unsqueeze(-1)
            q_values_out = q_values.unsqueeze(-1)
            mean_pred_out = mean_pred.unsqueeze(-1)

        return {
            "samples": samples_out,
            "quantiles": q_values_out,
            "mean": mean_pred_out,
            "quantile_levels": quantiles
        }

    def loss(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute loss for training or evaluation."""
        # For training, forward() returns the loss directly from the HF model
        return self.forward(x, y, **kwargs)

    def _eval_step(self, x: torch.Tensor, y: torch.Tensor, **kwargs) -> Tuple[Dict, torch.Tensor]:
        """Evaluation step."""
        y_pred = self.predict(x, **kwargs)
        
        # Compute metrics
        score_dict = {}
        if self.metrics:
            from quito.metrics import cal_score
            for metric in self.metrics:
                score = cal_score(metric, y_pred=y_pred, y_true=y[:, -self.forecast_horizon:, :])
                score_dict[metric] = score
        
        return score_dict, y_pred
