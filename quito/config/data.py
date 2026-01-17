"""
Data configuration classes for QUITO library.
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from omegaconf import DictConfig

from quito.config.base import BaseConfig

class DatasetType(Enum):
    """
    Enumeration of supported dataset types in QuitoBench.
    
    Defines the various data formats and sources that can be loaded
    and processed by the framework.
    """
    SYNTHETIC = "synthetic"
    UCR = "ucr"
    UEA = "uea"
    CUSTOM = "custom"
    CSV = "csv"
    JSON = "json"
    NPZ = "npz"
    HDF5 = "hdf5"


class Freq(Enum):
    """
    Enumeration of supported time series frequency types.
    
    Represents the temporal frequency/granularity of time series data.
    """
    M = "M"  # Monthly
    H = "H"  # Hourly


class Features(Enum):
    """
    Enumeration of feature type configurations for time series models.
    
    Defines how features are handled in the model:
    - M: Multivariate input -> Multivariate output
    - S: Univariate input -> Univariate output  
    - MS: Multivariate input -> Univariate output (selective forecasting)
    """
    M = "M"
    S = "S"
    MS = "MS"
    

@dataclass
class DatasetConfig(BaseConfig):
    """
    Configuration for individual dataset instances.
    
    Defines parameters for a single dataset including file location,
    train/validation/test splits, preprocessing options, and dataset class.
    
    Attributes:
        train_ratio (float): Proportion of data for training (0.0 to 1.0).
        valid_ratio (float): Proportion of data for validation (0.0 to 1.0).
        test_ratio (float): Proportion of data for testing (0.0 to 1.0).
        file_name (str): Name of the data file (relative to data_dir).
        is_pretrain (bool): Whether this dataset is for pre-training.
        freq (Freq): Time series frequency (e.g., hourly, monthly).
        ds_cls (str): Name of the dataset class to use for loading.
        ds_kwargs (dict): Additional keyword arguments for dataset class.
        target (str, optional): Target column name for multivariate->univariate.
        anomaly_remove_method (str, optional): Method for anomaly removal.
        na_impute_method (str): Method for missing value imputation.
            Defaults to 'linear'.
        ids (List[str], optional): List of specific time series IDs to load.
    """
    train_ratio: float 
    valid_ratio: float
    test_ratio: float
    file_name: str
    is_pretrain: bool
    freq: Freq
    ds_cls: str
    ds_kwargs: dict = field(default_factory=dict)
    target: str = None
    anomaly_remove_method: str = None
    na_impute_method: str = 'linear'
    ids: List[str] = None

    def split(self, L, test=True):
        """
        Calculate train/validation/test split sizes from total length.
        
        Computes the number of samples for each split based on the configured
        ratios and total data length.
        
        Args:
            L (int): Total length of the time series data.
            test (bool, optional): Whether to include test split.
                If False, only returns train and validation sizes.
                Defaults to True.
        
        Returns:
            Tuple[int, int, int]: A tuple containing:
                - train_size: Number of training samples
                - valid_size: Number of validation samples
                - test_size: Number of test samples (0 if test=False)
                
        Example:
            >>> config = DatasetConfig(train_ratio=0.7, valid_ratio=0.2, test_ratio=0.1, ...)
            >>> train, val, test = config.split(1000, test=True)
            >>> # Returns: (700, 200, 100)
        """
        # data: (N, L, C)
        if not test:
            valid_size = int(L * self.valid_ratio)
            train_size = L - valid_size
            return train_size, valid_size, 0
        else:
            train_size = int(L * self.train_ratio)
            valid_size = int(L * self.valid_ratio)
            test_size = L - train_size - valid_size
        
            return train_size, valid_size, test_size
    
    def get_ds_class(self, dataset_mapping):
        """
        Get the dataset class from the mapping registry.
        
        Looks up the dataset class by name in the provided mapping dictionary.
        This enables dynamic dataset class selection based on configuration.
        
        Args:
            dataset_mapping (dict): Dictionary mapping dataset class names
                to actual dataset classes (e.g., {'TimeSeriesDataset': TimeSeriesDataset}).
        
        Returns:
            type: The dataset class to use for loading this dataset.
            
        Raises:
            AssertionError: If the dataset class name is not found in the mapping.
            
        Example:
            >>> from quito.datasets import DATASET_MAPPING
            >>> dataset_class = config.get_ds_class(DATASET_MAPPING)
            >>> dataset = dataset_class(...)
        """
        ds_cls = dataset_mapping.get(self.ds_cls)
        assert ds_cls is not None, f"dataset class {self.ds_cls} not found in dataset_mapping"
        
        return ds_cls

    def validate(self):
        """
        Validate dataset configuration parameters.
        
        Checks that all split ratios are valid (between 0 and 1) and that
        they sum to approximately 1.0 (within 0.01 tolerance).
        
        Raises:
            ValueError: If any ratio is outside [0, 1] or if ratios don't sum to 1.0.
        """
        if not 0 <= self.train_ratio <= 1:
            raise ValueError("train_split must be between 0 and 1")
        
        if not 0 <= self.valid_ratio <= 1:
            raise ValueError("val_split must be between 0 and 1")
        
        if not 0 <= self.test_ratio <= 1:
            raise ValueError("test_split must be between 0 and 1")
        
        # Check that splits sum to approximately 1
        total_split = self.train_ratio + self.valid_ratio + self.test_ratio
        if abs(total_split - 1.0) > 0.01:
            raise ValueError(f"Data splits must sum to 1.0, got {total_split}")
        
        
@dataclass
class DataConfig(BaseConfig):
    """
    Main configuration class for time series data loading and preprocessing.
    
    This class defines all parameters needed for loading, preprocessing, and
    batching time series datasets. It aggregates multiple DatasetConfig objects
    and provides common settings for sequence length, forecast horizon, and
    data loading parameters.
    
    Attributes:
        data_dir (str): Base directory containing all dataset files.
        dataset_configs (dict): Dictionary of dataset-specific configurations.
            Each key maps to a DatasetConfig dictionary.
        decoder_label_len (int, optional): Extra label length for decoder input
            in encoder-decoder frameworks. Defaults to 24.
        seq_len (int, optional): Input sequence length (lookback window).
            Defaults to 100.
        forecast_horizon (int, optional): Prediction horizon length.
            Defaults to 24.
        features (str, optional): Feature type configuration. Options:
            - "M": Multivariate -> Multivariate
            - "S": Univariate -> Univariate
            - "MS": Multivariate -> Univariate (selective)
            Defaults to "MS".
        normalize (bool, optional): Whether to normalize time series data.
            Defaults to True.
        global_test_point (str, optional): Global test point timestamp for
            temporal splitting. Format: 'YYYY-MM-DD HH:MM:SS'. Defaults to '2024-09-01 00:00:00'.
        batch_size (int, optional): Batch size for data loading. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle training data. Defaults to True.
        num_workers (int, optional): Number of data loading workers. Defaults to 4.
        pin_memory (bool, optional): Whether to pin memory for faster GPU transfer.
            Defaults to True.
    """
    # Dataset identification
    data_dir: str
    dataset_configs: dict # 每个数据集的config
    decoder_label_len: int = 24  # extra label len to the decoder for encoder-decoder framework
    seq_len: int = 100 # the prediction length for the model
    forecast_horizon: int = 24
    features: str = "MS"  # M: multivariate->multivariate, S: univariate->univariate, MS: multivariate->univariate
    normalize: bool = True
    global_test_point: str = '2024-09-01 00:00:00'
    
    # Data loading
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True
    
    def __post_init__(self):
        self.ds_config_dict = {}
        for k, v in self.dataset_configs.items():
            v['freq'] = Freq(v['freq'])
            self.ds_config_dict[k] = DatasetConfig(**v)
        
        super().__post_init__()
    
    def validate(self):
        """
        Validate data configuration parameters.
        
        Ensures that all critical parameters are positive and within valid ranges.
        
        Raises:
            ValueError: If seq_len, forecast_horizon, or batch_size are <= 0.
        """
        if self.seq_len <= 0:
            raise ValueError("sequence_length must be positive")
        
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

def create_data_config(config: DictConfig):
    """
    Create a DataConfig instance from an OmegaConf configuration dictionary.
    
    Extracts data-related configuration from the main config object, combining
    settings from 'data.common', 'data.datasets', and 'training' sections.
    
    Args:
        config (DictConfig): OmegaConf configuration dictionary containing:
            - data.common: Common data settings (data_dir, seq_len, etc.)
            - data.datasets: Dictionary of dataset-specific configs
            - training: Training settings (batch_size, num_workers, etc.)
    
    Returns:
        DataConfig: Configured DataConfig instance ready for use.
        
    Raises:
        KeyError: If required configuration sections are missing.
        
    Example:
        >>> from omegaconf import OmegaConf
        >>> config = OmegaConf.load("config.yaml")
        >>> data_config = create_data_config(config)
    """
    common_config = config.data.common
    dataset_configs = config.data.datasets
    training_config = config.training
    data_config = DataConfig(data_dir=common_config.data_dir,
                             seq_len=common_config.seq_len,
                             decoder_label_len=common_config.decoder_label_len,
                             forecast_horizon=common_config.forecast_horizon,
                             features=common_config.features,
                             normalize=common_config.normalize,
                             batch_size=training_config.batch_size,
                             pin_memory=training_config.pin_memory,
                             num_workers=training_config.num_workers,
                             shuffle=training_config.shuffle,
                             dataset_configs=dataset_configs,
                             )
    
    return data_config
    