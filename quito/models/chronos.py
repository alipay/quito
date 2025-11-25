"""
Chronos Model Wrapper - Simple and lightweight for time series forecasting.
"""
import logging
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
            
            # ChronosPipeline.model returns a ChronosModel wrapper, not the raw T5 model
            # We need to access the underlying T5 model for training
            # Try to find it now (will retry in forward() if needed)
            self.t5_model = None
            
            # Debug: Log model structure for troubleshooting
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                logging.debug(f"ChronosModel type: {type(self.model)}")
                logging.debug(f"ChronosModel attributes: {[a for a in dir(self.model) if not a.startswith('_')][:20]}")
                logging.debug(f"T5 model type: {type(self.t5_model)}")
                if hasattr(self.model, '__dict__'):
                    logging.debug(f"ChronosModel __dict__ keys: {list(self.model.__dict__.keys())[:10]}")
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

    def _find_t5_model(self):
        """
        Helper method to find the underlying T5 model in ChronosModel.
        Returns the T5 model or None if not found.
        """
        chronos_model = self.model
        
        # Try direct attributes
        for attr_name in ['model', 'transformer', 't5_model', 'backbone', 'encoder_decoder', 'base_model']:
            if hasattr(chronos_model, attr_name):
                candidate = getattr(chronos_model, attr_name)
                # Check if it looks like a HuggingFace T5 model
                if hasattr(candidate, 'config') and hasattr(candidate.config, 'model_type'):
                    if candidate.config.model_type == 't5':
                        return candidate
                # Or if it has forward and accepts input_ids
                if hasattr(candidate, 'forward'):
                    # Try to inspect forward signature (basic check)
                    import inspect
                    try:
                        sig = inspect.signature(candidate.forward)
                        if 'input_ids' in sig.parameters:
                            return candidate
                    except:
                        pass
        
        # Try accessing through __dict__
        if hasattr(chronos_model, '__dict__'):
            for key, value in chronos_model.__dict__.items():
                if hasattr(value, 'config') and hasattr(value.config, 'model_type'):
                    if value.config.model_type == 't5':
                        return value
        
        return None

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Forward pass for training.
        
        Args:
            x: Context time series [batch, seq_len, 1]
            y: Target time series [batch, pred_len, 1]
            **kwargs: Additional arguments (filtered to avoid conflicts)
            
        Returns:
            Loss tensor for training
        """
        # Filter out any keys that might conflict (like 'labels' from batch)
        # These are not needed for Chronos forward pass
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['labels', 'x_mark', 'y_mark']}
        
        if y is None:
            # If no target provided, assume inference/generation
            return self.predict(x, **filtered_kwargs)
             
        # Prepare data for training (fine-tuning)
        if x.dim() == 3:
            x = x.squeeze(-1)  # [batch, seq_len]
        if y.dim() == 3:
            y = y.squeeze(-1)  # [batch, pred_len]
        
        # Ensure tensors are on correct device
        x = x.to(self.device)
        y = y.to(self.device)
            
        # Tokenize context (x) and target (y)
        # Chronos tokenizer converts float time series to token IDs
        try:
            # Try the standard Chronos tokenizer API
            if hasattr(self.tokenizer, 'context_input_transform'):
                context_ids, context_mask, _ = self.tokenizer.context_input_transform(x)
                target_ids, target_mask, _ = self.tokenizer.context_input_transform(y)
            elif hasattr(self.tokenizer, 'encode'):
                # Alternative API: direct encode method
                context_ids = self.tokenizer.encode(x)
                target_ids = self.tokenizer.encode(y)
                context_mask = torch.ones_like(context_ids, dtype=torch.bool)
                target_mask = torch.ones_like(target_ids, dtype=torch.bool)
            else:
                # Fallback: use pipeline's internal tokenization
                # Concatenate context and target for sequence-to-sequence training
                # Format: [context; target] -> predict [target]
                full_sequence = torch.cat([x, y], dim=1)  # [batch, seq_len + pred_len]
                
                # Use pipeline's predict method in training mode (if supported)
                # Otherwise, we'll need to tokenize manually
                raise NotImplementedError(
                    "Chronos tokenizer API not recognized. "
                    "Please ensure you have the latest version of chronos-forecasting library, "
                    "or use the pipeline's training methods directly."
                )
        except Exception as e:
            # If tokenization fails, provide helpful error
            raise RuntimeError(
                f"Failed to tokenize inputs for Chronos training: {e}\n"
                "Chronos training requires proper tokenization of time series data.\n"
                "Please check:\n"
                "  1. You have the latest chronos-forecasting library installed\n"
                "  2. The tokenizer API matches the expected format\n"
                "  3. Your input shapes are correct: x=[batch, seq_len], y=[batch, pred_len]"
            ) from e
        
        # Ensure tokenized inputs are on correct device
        context_ids = context_ids.to(self.device)
        context_mask = context_mask.to(self.device)
        target_ids = target_ids.to(self.device)
        
        # Forward pass through T5 model
        # T5 expects: input_ids, attention_mask, labels
        # Labels are automatically shifted right for decoder training in HF T5
        # Use the underlying T5 model (not the ChronosModel wrapper)
        try:
            # Get the actual T5 model (from self.t5_model set in __init__)
            t5_model = getattr(self, 't5_model', None)
            
            # If t5_model wasn't found or is still the ChronosModel wrapper, try to find it
            if t5_model is None or t5_model is self.model or type(t5_model).__name__ == 'ChronosModel':
                # Use helper method to find T5 model
                t5_model = self._find_t5_model()
                if t5_model is None:
                    # If still not found, this is a problem
                    raise RuntimeError(
                        "Could not find underlying T5 model in ChronosModel.\n"
                        f"ChronosModel type: {type(self.model)}\n"
                        f"ChronosModel attributes: {[a for a in dir(self.model) if not a.startswith('_')][:20]}\n"
                        "Please check the Chronos library documentation for how to access the T5 model.\n"
                        "You may need to update the chronos-forecasting library or use a different training approach."
                    )
                else:
                    # Cache it for future use
                    self.t5_model = t5_model
                    logging.info(f"Found T5 model: {type(t5_model).__name__}")
            
            # Safety check: ensure we're not accidentally calling ourselves
            if t5_model is self:
                raise RuntimeError(
                    "Error: t5_model points to ChronosModel itself. "
                    "This indicates a bug in model initialization."
                )
            
            # If model is wrapped (e.g., in DDP), unwrap it
            if hasattr(t5_model, 'module'):
                t5_model = t5_model.module
            
            # Verify we have a T5-like model (should have forward method that accepts input_ids, labels)
            if not hasattr(t5_model, 'forward'):
                raise RuntimeError(
                    f"Model does not have 'forward' method. Model type: {type(t5_model)}"
                )
            
            # Call T5 model forward with proper arguments
            # Use __call__ (which is what () does) - HF models use this for preprocessing
            outputs = t5_model(
                input_ids=context_ids,
                attention_mask=context_mask,
                labels=target_ids
            )
            
            # Return loss for training
            if hasattr(outputs, 'loss'):
                return outputs.loss
            else:
                # If no loss attribute, compute it manually
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                # Compute cross-entropy loss manually if needed
                # For now, raise error to debug
                raise ValueError(f"Model output has no 'loss' attribute. Output type: {type(outputs)}")
        except (TypeError, AttributeError) as e:
            # If there's an error, provide helpful debugging info
            error_msg = (
                f"Error calling T5 model forward: {e}\n"
                f"ChronosModel type: {type(self.model)}\n"
                f"T5 model type: {type(getattr(self, 't5_model', None))}\n"
                f"ChronosModel attributes: {[a for a in dir(self.model) if not a.startswith('_')][:15]}\n"
                "\n"
                "Troubleshooting:\n"
                "1. Check if ChronosModel has a 'model', 'transformer', or 't5_model' attribute\n"
                "2. The Chronos library structure may have changed\n"
                "3. You may need to use ChronosPipeline's training methods directly\n"
                "4. Try: print(dir(pipeline.model)) to see available attributes"
            )
            raise RuntimeError(error_msg) from e

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
        # Filter out any unexpected keys from kwargs that might cause issues
        # Remove keys that are not needed for forward pass
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['labels', 'x_mark', 'y_mark']}
        # For training, forward() returns the loss directly from the HF model
        return self.forward(x, y, **filtered_kwargs)

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
