"""
Model configuration classes for QuitoBench library.

This module provides configuration classes for all time series forecasting models
supported by QuitoBench. If pretrained_model_name_or_path is specified, the model
will be loaded from HuggingFace, and the configuration will be of type
PretrainedConfig instead of ModelConfig.

Model configurations follow a naming convention: {ModelName}ModelConfig, where
ModelName matches the model_name field in the YAML configuration file.
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
    Base configuration class for time series forecasting models.
    
    This class provides common configuration parameters for all time series models
    in QuitoBench. Model-specific configurations inherit from this class and add
    their own parameters. The class uses a registry pattern for automatic discovery.
    
    Naming convention: Model configs are named as {ModelName}ModelConfig, where
    ModelName matches the model_name field in the YAML configuration file.
    
    Attributes:
        REGISTRY (ClassVar[Dict]): Class-level registry of all ModelConfig subclasses.
        model_name (str): Name of the model (e.g., 'PatchTST', 'CrossFormer').
        pretrained_model_name_or_path (Optional[str]): Path to pretrained model.
            If provided, loads from HuggingFace or local path. Defaults to None.
    """
    REGISTRY: ClassVar[Dict[str, Type["ModelConfig"]]] = {}
    
    # Model identification
    model_name: str = 'PatchTST'
    pretrained_model_name_or_path: Optional[str] = None # This is a placeholder for model from huggingface
    # Data dimensions
    input_dim: int = 5
    hidden_dim: int = 64
    output_dim: int = 5
    
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
    # this is might be different from input_dim and output_dim, this will be used automatrically in model
    enc_in: int = 5  # Encoder input size
    dec_in: int = 1  # Decoder input size
    c_out: int = 5 # Output size

    checkpoint_path: str = None

    def validate(self):
        """
        Validate model configuration parameters.
        
        This method should be overridden in subclasses to implement
        model-specific validation logic.
        
        Raises:
            ValueError: If any configuration parameter is invalid.
        """
        pass
    
    @classmethod
    def get_default_config(cls, model_name: str) -> "ModelConfig":
        """
        Get default configuration for a specific model type.
        
        Provides sensible default parameters for a given model architecture.
        Useful for quick experimentation and as a starting point for configuration.
        
        Args:
            model_name (str): Name of the model type (e.g., 'PatchTST', 'CrossFormer').
        
        Returns:
            ModelConfig: Default configuration instance for the specified model.
            
        Raises:
            NotImplementedError: Always raised in base class. Subclasses should
                implement this method with model-specific defaults.
        """
        raise NotImplementedError
    
    def to_huggingface_config(self) -> Dict[str, Any]:
        """
        Convert to Hugging Face Transformers compatible configuration dictionary.
        
        Converts QuitoBench model configuration to a format compatible with
        Hugging Face Transformers, enabling interoperability with HF models.
        
        Returns:
            Dict[str, Any]: Dictionary with keys matching Hugging Face model
                configuration parameters.
                
        Raises:
            NotImplementedError: Always raised in base class. Subclasses should
                implement this method for Hugging Face compatibility.
        """
        raise NotImplementedError
    
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        ModelConfig.REGISTRY[cls.__name__] = cls
    

@dataclass
class PatchTSTModelConfig(ModelConfig):
    """
    Configuration for PatchTST (Patch-based Time Series Transformer) model.
    
    PatchTST divides time series into patches and applies transformer attention
    to learn temporal patterns. It's designed for efficient long-term forecasting.
    
    Key parameters:
    - patch_len: Length of each patch
    - stride: Stride for patch extraction
    - e_layers: Number of encoder layers
    - revin: Whether to use RevIN normalization
    
    Reference:
        Nie et al. (2023). "A Time Series is Worth 64 Words: Long-term Forecasting
        with Transformers"
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
    """
    Configuration for DLinear model.
    
    DLinear is a simple linear model that serves as a strong baseline for
    time series forecasting. It uses moving average decomposition and linear layers.
    
    Attributes:
        kernel_size (int, optional): Kernel size for moving average.
            Defaults to 25.
        individual (bool, optional): Whether to use individual linear layers
            per channel. Defaults to False.
        revin (bool, optional): Whether to use RevIN normalization.
            Defaults to True.
            
    Reference:
        Zeng et al. (2023). "Are Transformers Effective for Time Series Forecasting?"
    """
    kernel_size: int = 25
    individual: bool = False
    revin: bool = True


@dataclass
class TSTransformerModelConfig(ModelConfig):
    """
    Configuration for TSTransformer (Time Series Transformer) model.
    
    TSTransformer applies transformer architecture to time series with alternating
    time and feature attention layers to capture both temporal and cross-feature
    dependencies.
    
    Attributes:
        layers (list): List of layer types alternating between 'time_full' and
            'feature_full' attention.
        patch_size (int): Size of patches for patch-based processing.
        time_pe_type (str, optional): Type of positional encoding for time dimension.
        feature_pe_type (str): Type of positional encoding for feature dimension.
            Defaults to 'subspace'.
        d_model (int): Model dimension. Defaults to 512.
        d_ff (int): Feed-forward dimension. Defaults to 1024.
        n_heads (int): Number of attention heads. Defaults to 8.
        revin (bool): Whether to use RevIN normalization. Defaults to True.
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
    """
    Configuration for TiRex (Time Series Representation Learning) model.
    
    TiRex is a pre-trained foundation model for time series forecasting that
    supports zero-shot inference. Configuration is typically loaded from
    pretrained model checkpoints.
    """
    pass


@dataclass
class ITransformerModelConfig(ModelConfig):
    """
    Configuration for iTransformer model.
    
    iTransformer inverts the standard transformer architecture by treating
    variates as tokens and time points as features, enabling better cross-variate
    dependency learning.
    
    Attributes:
        output_attention (bool): Whether to output attention weights.
            Defaults to False.
        e_layers (int): Number of encoder layers. Defaults to 3.
        use_norm (bool): Whether to use layer normalization. Defaults to True.
        d_model (int): Model dimension. Defaults to 512.
        d_ff (int): Feed-forward dimension. Defaults to 512.
        embed (str): Embedding type. Defaults to 'timeF'.
        freq (str): Frequency string. Defaults to 'h' (hourly).
        dropout (float): Dropout rate. Defaults to 0.1.
        activation (str): Activation function. Defaults to 'gelu'.
        class_strategy (str): Classification strategy. Defaults to 'projection'.
        factor (float): Scaling factor. Defaults to 1.0.
        n_heads (int): Number of attention heads. Defaults to 8.
        
    Reference:
        Liu et al. (2024). "iTransformer: Inverted Transformers Are Effective
        for Time Series Forecasting"
    """
    output_attention: bool = False
    e_layers: int = 3
    use_norm: bool = True
    d_model: int = 512
    d_ff: int = 512
    embed: str = 'timeF'
    freq: str = 'h'
    dropout: float = 0.1
    activation: str = 'gelu'
    class_strategy: str = 'projection'
    factor: float = 1.0
    n_heads: int = 8


@dataclass
class TSMixerModelConfig(ModelConfig):
    """
    Configuration for TSMixer model.
    
    TSMixer is an MLP-based model that uses mixing layers to capture temporal
    and cross-feature dependencies. It's a lightweight alternative to transformers.
    
    Attributes:
        num_blocks (int): Number of mixing blocks. Defaults to 2.
        d_ff (int): Feed-forward dimension. Defaults to 64.
        norm_type (str): Normalization type. Defaults to 'layer'.
        revin (bool): Whether to use RevIN normalization. Defaults to True.
        
    Reference:
        Ekambaram et al. (2023). "TSMixer: Lightweight MLP-Mixer Model for
        Multivariate Time Series Forecasting"
    """
    num_blocks: int = 2
    d_ff: int = 64
    norm_type: str = 'layer'
    revin: bool = True


@dataclass
class CrossFormerModelConfig(ModelConfig):
    """
    Configuration for CrossFormer model.
    
    CrossFormer uses cross-dimension attention to capture dependencies across
    both time and feature dimensions in multivariate time series.
    
    Attributes:
        seg_len (int): Segment length for patch-based processing. Defaults to 12.
        d_model (int): Model dimension. Defaults to 64.
        d_ff (int): Feed-forward dimension. Defaults to 128.
        n_heads (int): Number of attention heads. Defaults to 2.
        win_size (int): Window size for segment merging. Defaults to 4.
        factor (int): Factor for cross-dimension attention. Defaults to 10.
        e_layers (int): Number of encoder layers. Defaults to 3.
        dropout (float): Dropout rate. Defaults to 0.0.
        revin (bool): Whether to use RevIN normalization. Defaults to True.
        baseline (bool): Whether to use baseline (mean) prediction.
            Defaults to False.
            
    Reference:
        Zhang & Yan (2022). "Crossformer: Transformer Utilizing Cross-Dimension
        Dependency for Multivariate Time Series Forecasting"
    """
    seg_len: int = 12
    d_model: int = 64
    d_ff: int = 128
    n_heads: int = 2
    win_size: int = 4
    factor: int = 10
    e_layers: int = 3
    dropout: float = 0.0
    revin: bool = True
    baseline: bool = False


@dataclass
class PyraFormerModelConfig(ModelConfig):
    """
    Configuration for PyraFormer model.
    
    PyraFormer uses pyramidal attention with multi-scale windows to capture
    both short-term and long-term dependencies in time series.
    
    Attributes:
        decoder (str): Decoder type (fixed to 'FC'). Defaults to 'FC'.
        d_model (int): Model dimension. Defaults to 512.
        window_size (Union[List[int], str]): Window sizes for pyramidal attention.
            Can be a list or comma-separated string. Defaults to [4, 4, 4].
        truncate (bool): Whether to truncate sequences. Defaults to False.
        inner_size (int): Inner size parameter. Defaults to 3.
        use_tvm (bool): Whether to use TVM optimization. Defaults to False.
        d_inner_hid (int): Inner hidden dimension. Defaults to 512.
        n_head (int): Number of attention heads. Defaults to 6.
        d_k (int): Key dimension. Defaults to 128.
        d_v (int): Value dimension. Defaults to 128.
        n_layer (int): Number of layers. Defaults to 2.
        embed_type (str): Embedding type (fixed to 'DataEmbedding').
            Defaults to 'DataEmbedding'.
        CSCM (str): CSCM module type. Defaults to 'Bottleneck_Construct'.
        d_bottleneck (int): Bottleneck dimension. Defaults to 128.
        device (Optional[str]): Device string (set automatically in model).
            Defaults to None.
        revin (bool): Whether to use RevIN normalization. Defaults to True.
        
    Reference:
        Liu et al. (2021). "Pyraformer: Low-Complexity Pyramidal Attention
        for Long-Range Time Series Modeling and Forecasting"
    """
    decoder: str = 'FC' # fixed here
    d_model: int = 512
    window_size: Union[List[int], str] = field(default_factory=lambda: [4, 4, 4])
    truncate: bool = False
    inner_size: int = 3
    use_tvm: bool = False
    d_inner_hid: int = 512
    n_head: int = 6
    d_k: int = 128
    d_v: int = 128
    n_layer: int = 2
    embed_type: str = 'DataEmbedding' # fixed here
    CSCM: str = 'Bottleneck_Construct'
    d_bottleneck: int = 128
    device: Optional[str] = None # set up in the model
    revin: bool = True

    def __post_init__(self):
        """
        Post-initialization processing.
        
        Converts window_size from string to list if provided as comma-separated string.
        """
        if isinstance(self.window_size, str):
            self.window_size = [int(x) for x in self.window_size.split(',')]


@dataclass
class ESModelConfig(ModelConfig):
    """
    Configuration for ETS model.
    """
    trend: Optional[str] = None
    seasonal: Optional[str] = None
    seasonal_periods: Optional[int] = None
    damped_trend: bool = False


@dataclass
class NaiveForecasterModelConfig(ModelConfig):
    """
    Configuration for NaiveForecaster model.
    """
    method: str = 'mean'
    seasonal_period: int = 6