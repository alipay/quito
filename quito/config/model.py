"""
Model configuration classes for QUITO library. If pretrained_model_name_or_path is given, the model will be loaded from huggingface, and the output class wil
have type PretrainedConfig.
"""

from typing import Optional, List, Union, Dict, Any, ClassVar, Type
from dataclasses import dataclass, field
from enum import Enum
from omegaconf import DictConfig
from transformers import PretrainedConfig

from quito.config.base import BaseConfig
from quito.utils.common import register_to_mapping


@dataclass
class ModelConfig(BaseConfig):
    """
    Configuration for time series models, model config is named in special format {ModelName}ModelConfig,
    the ModelName will be model_name field in the config yaml.
    """
    REGISTRY: ClassVar[Dict[str, Type["ModelConfig"]]] = {}

    # Model identification
    model_name: str = 'PatchTST'
    pretrained_model_name_or_path: Optional[str] = None # This is a placeholder for model from huggingface
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

    def validate(self):
        pass

    @classmethod
    def get_default_config(cls, model_name: str) -> "ModelConfig":
        """
        Get default configuration for a specific model type.

        Args:
            model_name: Type of model to get default config for

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

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelConfig.REGISTRY[cls.__name__] = cls


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
class TSTransformerModelConfig(ModelConfig):
    """
    Configuration for TSTransformer model.
    """
    layers: list = field(default_factory=lambda: [
        'time_full',
        'feature_full',
        'time_full',
        'feature_full',
        'time_full',
        'feature_full',
        'time_full',
        'feature_full',
        'time_full',
        'feature_full',
        'time_full',
        'feature_full'])

    patch_size: int = 8
    time_pe_type: str = None
    feature_pe_type: str = 'subspace'
    d_model: int = 512
    d_ff: int = 1024
    act: str = 'GELU'
    n_heads: int = 8
    num_groups: int = None
    d_k: int = None
    d_v: int = None
    attn_dropout: float = 0.0
    pre_norm: bool = True
    norm_type: str = 'LayerNorm'
    rope: bool = False
    dropout: float = 0.0
    max_context_len: int = 5000
    max_features: int = 10
    revin: bool = True


@dataclass
class TiRexModelConfig(ModelConfig):
    pass
