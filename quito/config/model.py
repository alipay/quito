"""
Model configuration classes for QUITO library.
"""

from typing import Optional, List, Union, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from omegaconf import DictConfig

from quito.config.base import BaseConfig
from quito.utils.common import register_to_mapping


class ModelType(Enum):
    """Enumeration of supported model types."""
    PATCHTST = 'patchtst'
    DLINEAR = 'dlinear'
    CHRONOS = 'chronos'
    MORIAI = 'moriai'
    HUGGINGFACE = 'huggingface'

@dataclass
class ModelConfig(BaseConfig):
    """
    Configuration for time series models.
    
    This class defines the parameters for different types of time series models,
    including GAN-based, Transformer-based, VAE-based, and ODE-based models.
    """
    
    # Model identification
    model_name: ModelType
    
    # Data dimensions
    input_dim: int = 1
    hidden_dim: int = 64
    output_dim: int = 1
    
    # Architecture parameters
    num_layers: int = 2
    dropout: float = 0.1
    activation: str = "relu"

    # Transformer-specific parameters
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    max_seq_len: int = 5000
    attention_dropout: float = 0.1
    pre_norm: bool = True
    pe: str = 'zero' # Positional embedding
    
    # Time series forecasting parameters
    seq_len: int = 512  # Input sequence length
    label_len: int = 48  # Label length for decoder start
    forecast_horizon: int = 192  # Prediction length
    decoder_label_len: int = 48  # Decoder input length
    enc_in: int = 1  # Encoder input size
    dec_in: int = 1  # Decoder input size
    c_out: int = 1  # Output size
    
    def __post_init__(self):
        """Post-initialization validation."""
        self.model_name = ModelType(self.model_name)
        super().__post_init__()

    def validate(self):
        pass
    
    @classmethod
    def get_default_config(cls, model_type: ModelType) -> "ModelConfig":
        """
        Get default configuration for a specific model type.
        
        Args:
            model_type: Type of model to get default config for
            
        Returns:
            Default configuration for the model type
        """
        raise NotImplementedError
    
    def to_huggingface_config(self) -> Dict[str, Any]:
        """
        Convert to Hugging Face Transformers compatible configuration.
        
        Returns:
            Dictionary compatible with Hugging Face Transformers
        """
        raise NotImplementedError
    

@dataclass
class PatchTSTModelConfig(ModelConfig):
    """
    PatchTST Model config, match the original implementation.
    """
    e_layers: int = 2
    n_heads: int = 8
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.05
    fc_dropout: float = 0.05
    attention_dropout: float = 0.0
    individual: bool = False
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = 'end'
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    decomposition: bool = False
    kernel_size: int = 25
    pre_norm: bool = True
    res_attention: bool = True
    pe: str = 'zero'
    learn_pe: bool = True
    pretrain_head: bool = False
    head_type: str = 'flatten'
    activation: str = 'gelu'
    head_dropout: float = 0.0
    d_k: int = None
    d_v: int = None
    norm: str = 'BatchNorm'


@dataclass
class DLinearModelConfig(ModelConfig):
    kernel_size: int = 25
    individual: bool = False
    enc_in: int = 7


@dataclass
class ChronosModelConfig(ModelConfig):
    """
    Configuration for Chronos model.
    """
    pretrained_model_name_or_path: str = "amazon/chronos-t5-small"
    prediction_length: int = 192
    num_samples: int = 20
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 1.0


@dataclass
class MoriaiModelConfig(ModelConfig):
    """
    Configuration for Moirai model.
    """
    pretrained_model_name_or_path: str = "Salesforce/moirai-1.0-R-small"
    prediction_length: int = 96
    patch_size: int = 64
    context_length: int = 1000
    num_samples: int = 100
    target_dim: int = 1  # Number of target variables
    feat_dynamic_real_dim: int = 0  # Number of dynamic real features
    past_feat_dynamic_real_dim: int = 0  # Number of past dynamic real features
    mode: str = "inference"  # Only "inference" is supported (pretrain is not available)

@dataclass
class HuggingFaceModelConfig(ModelConfig):
    """
    Configuration for generic Hugging Face model.
    """
    pretrained_model_name_or_path: str = "google/timesfm-1.0-200m"
    trust_remote_code: bool = True
