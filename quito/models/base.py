"""
Base model classes for QUITO library.
"""
import logging
import os
import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
from functools import partial
from transformers import PretrainedConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quito.config.model import ModelConfig
from quito.config.training import TrainerConfig
from quito.utils.common import set_seed, get_device
from quito.metrics import get_metric_fn, cal_score


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all time series forecasting models in QuitoBench.
    
    This class provides common functionality for model initialization, device
    management, saving/loading checkpoints, and basic training/evaluation methods.
    All model implementations in QuitoBench should inherit from this class.
    
    The class automatically registers subclasses in the REGISTRY dictionary,
    enabling automatic model discovery and instantiation.
    
    Attributes:
        REGISTRY (dict): Class-level dictionary that automatically registers
            all subclasses by their class name.
        config (ModelConfig): Model configuration object.
        local_rank (int): Local rank for distributed training (-1 for CPU).
        device (str): Device string ('cpu' or 'cuda:N').
        seq_len (int): Input sequence length.
        forecast_horizon (int): Prediction horizon length.
        decoder_label_len (int): Decoder label length for encoder-decoder models.
        metrics (list): List of evaluation metrics to compute.
        loss_fn (nn.Module): Loss function for training.
        
    Example:
        >>> class MyModel(BaseModel):
        ...     def forward(self, x, y=None, **kwargs):
        ...         return self.linear(x)
        ...     def loss(self, x, y, **kwargs):
        ...         return self.loss_fn(self.forward(x), y)
        ...     def _eval_step(self, x, y, **kwargs):
        ...         return {'mse': self.loss_fn(self.forward(x), y)}, self.forward(x)
    """
    REGISTRY = {} # register all subclasses
    
    def __init__(self, config: ModelConfig | PretrainedConfig, local_rank: int):
        """
        Initialize the model with configuration and device setup.
        
        Sets up the model with the provided configuration, determines the
        appropriate device (CPU or GPU), and initializes base attributes
        from the configuration.
        
        Args:
            config (ModelConfig | PretrainedConfig): Model configuration object
                containing architecture parameters and hyperparameters.
            local_rank (int): Local rank for distributed training. Use -1 for
                CPU mode, >=0 for GPU mode (specifies which GPU to use).
        """
        super().__init__()
        self.config = config
        self.local_rank = local_rank
        self.metrics = None  # set up in the trainer
        self.loss_fn = None # set up in the trainer

        # some base attributes setup
        self.seq_len = self.config.seq_len
        self.forecast_horizon = self.config.forecast_horizon
        self.decoder_label_len = self.config.decoder_label_len

        # Device management: handle CPU (-1) and GPU (>=0) cases
        self.device = f'cuda:{local_rank}' if local_rank >= 0 else 'cpu'

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Register the subclass by its name
        BaseModel.REGISTRY[cls.__name__] = cls

    def setup_loss_fn(self, loss_fn, loss_kwargs):
        """
        Set up the loss function for training.
        
        Configures the loss function based on the specified name and keyword
        arguments. Supported loss functions include MAE and MSE.
        
        Args:
            loss_fn (str): Name of the loss function. Options: 'mae', 'mse'.
            loss_kwargs (dict): Keyword arguments to pass to the loss function
                constructor.
                
        Raises:
            ValueError: If an unsupported loss function name is provided.
            
        Example:
            >>> model.setup_loss_fn('mse', {'reduction': 'mean'})
            >>> model.setup_loss_fn('mae', {})
        """
        logging.info(f'Loading loss function {loss_fn}')
        if loss_fn == 'mae':
            self.loss_fn = nn.L1Loss(**loss_kwargs)
        elif loss_fn == 'mse':
            self.loss_fn = nn.MSELoss(**loss_kwargs)
        else:
            raise ValueError(f'Loss function {loss_fn} not supported')

    @abstractmethod
    def forward(self, x, y=None, x_mark=None, y_mark=None, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        This method must be implemented by all subclasses. It performs the
        forward computation through the model architecture.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor for encoder-decoder models.
                Shape: (batch_size, forecast_horizon, n_features). Defaults to None.
            x_mark (torch.Tensor, optional): Time features for input sequence.
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features for target sequence.
                Defaults to None.
            **kwargs: Additional keyword arguments for model-specific parameters.
        
        Returns:
            torch.Tensor: Model output tensor. Shape depends on model architecture.
                Typically (batch_size, forecast_horizon, n_features) for forecasting.
        """
        pass
    
    @abstractmethod
    def loss(self, x, y, x_mark=None, y_mark=None, **kwargs) -> torch.Tensor:
        """
        Compute the loss for the model given input and target.
        
        This method must be implemented by all subclasses. It computes the
        training loss between model predictions and ground truth targets.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor): Target tensor of shape (batch_size, forecast_horizon, n_features).
            x_mark (torch.Tensor, optional): Time features for input sequence.
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features for target sequence.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        pass
    
    @abstractmethod
    def _eval_step(self, x, y, x_mark=None, y_mark=None, **kwargs) -> Tuple[Dict[str, torch.Tensor], Tuple]:
        """
        Perform a single evaluation step.
        
        This method must be implemented by all subclasses. It computes predictions
        and evaluation metrics for a batch of data during validation/testing.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor): Target tensor of shape (batch_size, forecast_horizon, n_features).
            x_mark (torch.Tensor, optional): Time features for input sequence.
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features for target sequence.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
                - score_dict: Dictionary mapping metric names to computed values
                - y_pred: Predicted tensor of shape (batch_size, forecast_horizon, n_features)
        """
        pass

    def save_pretrained(self, save_directory: Union[str, Path], **kwargs):
        """
        Save the model to a directory in a format compatible with HuggingFace-style loading.
        
        Saves the model state dict, configuration, and metadata to enable easy
        loading later. Creates the directory if it doesn't exist.
        
        Args:
            save_directory (Union[str, Path]): Directory path where the model
                will be saved. Will be created if it doesn't exist.
            **kwargs: Additional arguments (currently unused, for future extensions).
                
        Files created:
            - pytorch_model.bin: Model state dictionary
            - config.json: Model configuration
            - model_info.json: Model metadata (type, version, framework)
            
        Example:
            >>> model.save_pretrained("./saved_models/my_model")
            >>> # Creates: ./saved_models/my_model/pytorch_model.bin, config.json, etc.
        """
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), save_directory / "pytorch_model.bin")
        
        # Save configuration
        self.config.save(save_directory / "config.json")
        
        # Save model info
        model_info = {
            "model_type": self.__class__.__name__,
            "version": "0.1.0",
            "framework": "pytorch",
        }
        
        with open(save_directory / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, Path], **kwargs):
        """
        Load a pretrained model from a saved directory.
        
        Loads a model that was previously saved using save_pretrained(). The
        method loads the configuration, creates a model instance, and loads
        the trained weights.
        
        Args:
            pretrained_model_name_or_path (Union[str, Path]): Path to the directory
                containing the saved model files (pytorch_model.bin, config.json).
            **kwargs: Additional arguments passed to model initialization.
        
        Returns:
            BaseModel: Loaded model instance with pretrained weights.
            
        Raises:
            FileNotFoundError: If the model directory or required files don't exist.
            
        Example:
            >>> model = MyModel.from_pretrained("./saved_models/my_model")
            >>> # Loads model with pretrained weights
        """
        model_path = Path(pretrained_model_name_or_path)
        
        # Load configuration
        config = ModelConfig.from_file(model_path / "config.json")
        
        # Create model instance
        model = cls(config, **kwargs)
        
        # Load state dict with proper device handling
        state_dict = torch.load(
            model_path / "pytorch_model.bin",
            map_location=torch.device(model.device)
        )
        model.load_state_dict(state_dict)
        
        return model
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step on a batch of data.
        
        Sets the model to training mode, moves the batch to the appropriate device,
        performs forward pass, and computes the loss.
        
        Args:
            batch (Dict[str, torch.Tensor]): Dictionary containing input tensors.
                Expected keys include 'x' (input), 'y' (target), and optionally
                'x_mark', 'y_mark' (time features).
        
        Returns:
            torch.Tensor: Scalar loss tensor for backpropagation.
            
        Example:
            >>> batch = {'x': input_tensor, 'y': target_tensor}
            >>> loss = model.train_step(batch)
            >>> loss.backward()
        """
        self.train()
        
        # Move batch to device
        batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
        # Forward pass and loss computation
        loss = self.loss(**batch)
        
        return loss
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a single evaluation step on a batch of data.
        
        Sets the model to evaluation mode, disables gradient computation,
        and computes predictions and evaluation metrics.
        
        Args:
            batch (Dict[str, torch.Tensor]): Dictionary containing input tensors.
                Expected keys include 'x' (input), 'y' (target), and optionally
                'x_mark', 'y_mark' (time features).
        
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
                - score_dict: Dictionary mapping metric names to computed values
                - y_pred: Predicted tensor
                
        Example:
            >>> batch = {'x': input_tensor, 'y': target_tensor}
            >>> scores, predictions = model.eval_step(batch)
            >>> print(f"MSE: {scores['mse']}")
        """
        self.eval()
        
        with torch.no_grad():
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            score_dict, y_pred = self._eval_step(**batch)
            
            return score_dict, y_pred

    def generate(self, batch_size: int = 1, **kwargs) -> torch.Tensor:
        """
        Generate synthetic time series data (not implemented in base class).
        
        This method is intended for generative models that can sample new
        time series from the learned distribution. Most forecasting models
        do not implement this functionality.
        
        Args:
            batch_size (int, optional): Number of samples to generate.
                Defaults to 1.
            **kwargs: Additional generation parameters (model-specific).
        
        Returns:
            torch.Tensor: Generated time series tensor.
            
        Raises:
            NotImplementedError: Always raised in base class. Subclasses
                that support generation should override this method.
        """
        raise NotImplementedError("Generation not implemented for this model")
    
    def predict(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """
        Make predictions on input data.
        
        Convenience method for inference that sets the model to evaluation mode
        and disables gradient computation. Calls the forward() method internally.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor, optional): Target tensor for encoder-decoder models.
                Defaults to None.
            x_mark (torch.Tensor, optional): Time features for input sequence.
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features for target sequence.
                Defaults to None.
            **kwargs: Additional prediction parameters.
        
        Returns:
            torch.Tensor: Predicted tensor of shape (batch_size, forecast_horizon, n_features).
            
        Example:
            >>> predictions = model.predict(input_tensor)
            >>> # Returns: (batch_size, forecast_horizon, n_features)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x=x, y=y, x_mark=x_mark, y_mark=y_mark, **kwargs)

    def load(self, checkpoint_or_path: Union[str, dict]):
        """
        Load model weights from a checkpoint file or dictionary.
        
        Supports loading from checkpoint files saved by PyTorch trainers or
        from dictionaries containing state dictionaries. Handles different
        checkpoint formats (model_state_dict, state_dict).
        
        Args:
            checkpoint_or_path (Union[str, dict]): Path to checkpoint file (.pt, .pth)
                or dictionary containing the state dict.
        
        Raises:
            ValueError: If no valid state dictionary is found in the checkpoint.
            FileNotFoundError: If checkpoint file path doesn't exist.
            
        Example:
            >>> model.load("checkpoints/epoch_10.pt")
            >>> # Or from a dict:
            >>> checkpoint = torch.load("checkpoint.pt")
            >>> model.load(checkpoint)
        """
        if isinstance(checkpoint_or_path, str):
            ckpt = torch.load(checkpoint_or_path, map_location='cpu')
        else:
            ckpt = checkpoint_or_path

        if 'model_state_dict' in ckpt:
            self.load_state_dict(ckpt['model_state_dict'])
        elif 'state_dict' in ckpt:
            self.load_state_dict(ckpt['state_dict'])
        else:
            raise ValueError(f'No model state dict found in checkpoint {ckpt} !!')
        
        logging.info(f'Load model from checkpoint successfully')
    
    def predict_prob(self, x: torch.Tensor, quantiles: List[float] = [0.1, 0.5, 0.9], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate probabilistic forecasts with uncertainty quantification.
        
        This method is intended for models that can produce probabilistic
        predictions (e.g., quantile regression, distributional forecasting).
        Most point forecasting models do not implement this functionality.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, n_features).
            quantiles (List[float], optional): List of quantiles to compute.
                Common values: [0.1, 0.5, 0.9] for 10th, 50th, 90th percentiles.
                Defaults to [0.1, 0.5, 0.9].
            **kwargs: Additional prediction parameters.
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'samples': Monte Carlo samples of shape (batch, num_samples, pred_len, n_features)
                - 'quantiles': Quantile predictions of shape (batch, len(quantiles), pred_len, n_features)
                - 'mean': Mean prediction of shape (batch, pred_len, n_features)
                
        Raises:
            NotImplementedError: Always raised in base class. Subclasses that
                support probabilistic forecasting should override this method.
        """
        raise NotImplementedError


class TimeSeriesModel(BaseModel):
    """
    Base class for time series forecasting models.
    
    This class extends BaseModel with time series-specific functionality,
    including encoder-decoder input construction and standard loss/evaluation
    computation. Most QuitoBench forecasting models inherit from this class.
    
    The class provides default implementations of loss() and _eval_step() that
    work with encoder-decoder architectures, where decoder inputs are constructed
    from the target sequence.
    """
    
    def __init__(self, config: ModelConfig, local_rank=-1):
        """
        Initialize the time series model.
        
        Args:
            config (ModelConfig): Model configuration object.
            local_rank (int, optional): Local rank for distributed training.
                Defaults to -1 (CPU mode).
        """
        super().__init__(config, local_rank)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs):
        """
        Forward pass (must be implemented by subclasses).
        
        Args:
            x: Input tensor
            y: Target tensor (for encoder-decoder)
            x_mark: Input time features
            y_mark: Target time features
            **kwargs: Additional arguments
            
        Returns:
            Model output tensor
        """
        raise NotImplementedError

    def loss(self, x: torch.Tensor, y: torch.Tensor, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None,  **kwargs) -> torch.Tensor:
        """
        Compute the loss given a batch of inputs and targets.
        
        Constructs decoder inputs for encoder-decoder architectures and computes
        the loss between predictions and ground truth targets.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor): Target tensor of shape (batch_size, forecast_horizon, n_features).
            x_mark (torch.Tensor, optional): Time features for input sequence.
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features for target sequence.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        # construct decoder input for encoder-decoder framework
        x, y_in, x_mark, y_mark = self._construct_model_input(x, y, x_mark, y_mark)
        y_pred = self.forward(x=x, y=y_in, x_mark=x_mark, y_mark=y_mark, **kwargs)
        
        return self.loss_fn(y_pred, y[:, -self.forecast_horizon:, :])

    def _eval_step(self, x: torch.Tensor, y: torch.Tensor, x_mark: torch.Tensor = None, y_mark: torch.Tensor = None, **kwargs):
        """
        Perform a single evaluation step.
        
        Constructs decoder inputs, computes predictions, and evaluates metrics
        against ground truth targets. Handles both point predictions and quantile
        predictions if supported by the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor): Target tensor of shape (batch_size, forecast_horizon, n_features).
            x_mark (torch.Tensor, optional): Time features for input sequence.
                Defaults to None.
            y_mark (torch.Tensor, optional): Time features for target sequence.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        
        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor]: A tuple containing:
                - score_dict: Dictionary mapping metric names to computed values
                - y_pred: Point predictions of shape (batch_size, forecast_horizon, n_features)
        """
        # construct decoder input for encoder-decoder framework
        x, y_in, x_mark, y_mark = self._construct_model_input(x, y, x_mark, y_mark)
        y_pred = self.predict(x=x, y=y_in, x_mark=x_mark, y_mark=y_mark, **kwargs)
        score_dict = {}
        if isinstance(y_pred, tuple):
            # now it is a tuple of (y_pred_point, y_pred_quantile)
            y_pred_point, y_pred_quantile = y_pred
        else:
            y_pred_point = y_pred
            y_pred_quantile = None
            
        for metric in self.metrics:
            score = cal_score(metric_name=metric, y_pred=y_pred_point, y_true=y[:, -self.forecast_horizon:, :], x_train=x, y_pred_quantile=y_pred_quantile)
            score_dict[metric] = score

        return score_dict, y_pred_point
    
    def _construct_model_input(self, x, y, x_mark, y_mark):
        """
        Construct decoder input for encoder-decoder framework.
        
        Creates decoder input by concatenating the first decoder_label_len
        timesteps from the target sequence with zero-padded future timesteps.
        This is a common pattern in encoder-decoder architectures for time series.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_features).
            y (torch.Tensor): Target tensor of shape (batch_size, forecast_horizon, n_features).
            x_mark (torch.Tensor, optional): Time features for input sequence.
            y_mark (torch.Tensor, optional): Time features for target sequence.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - x: Original input tensor
                - dec_in: Decoder input tensor of shape (batch_size, decoder_label_len + forecast_horizon, n_features)
                - x_mark: Input time features (unchanged)
                - y_mark: Target time features (unchanged)
        """            
        dec_in = torch.zeros_like(y[:, -self.forecast_horizon:, :])
        dec_in = torch.cat([y[:, :self.decoder_label_len, :], dec_in], dim=1).float().to(self.device)

        return x, dec_in, x_mark, y_mark
        