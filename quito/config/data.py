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
    """Enumeration of supported dataset types."""
    SYNTHETIC = "synthetic"
    UCR = "ucr"
    UEA = "uea"
    CUSTOM = "custom"
    CSV = "csv"
    JSON = "json"
    NPZ = "npz"
    HDF5 = "hdf5"


class Freq(Enum):
    """Enumeration of supported frequency types."""
    M = "M"
    H = "H"


class Features(Enum):
    """Enumeration of supported feature types."""
    M = "M"
    S = "S"
    MS = "MS"
    

@dataclass
class DatasetConfig(BaseConfig):
    """
    The configration for the datasets
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
        split length L into train_size, valid_size and test_size
        """
        # data: (N, L, C)
        if not test:
            # if test is not needed, split the L into train and valid using valid_ratio
            valid_size = int(L * self.valid_ratio)
            train_size = L - valid_size
            return train_size, valid_size, 0
        else:
            train_size = int(L * self.train_ratio)
            valid_size = int(L * self.valid_ratio)
            test_size = L - train_size - valid_size

            return train_size, valid_size, test_size

    def get_ds_class(self, dataset_mapping):
        ds_cls = dataset_mapping.get(self.ds_cls)
        assert ds_cls is not None, f"dataset class {self.ds_cls} not found in dataset_mapping"

        return ds_cls

    def validate(self):
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
    Configuration for time series datasets.

    This class defines the parameters for loading and preprocessing
    time series data, including data sources, preprocessing steps,
    and augmentation strategies.
    """
    # Dataset identification
    data_dir: str
    dataset_configs: dict # 每个数据集的config
    decoder_label_len: int = 24  # extra label len to the decoder for encoder-decoder framework
    seq_len: int = 100 # the prediction length for the model
    forecast_horizon: int = 24
    features: str = "MS"  # M: multivariate->multivariate, S: univariate->univariate, MS: multivariate->univariate
    normalize: bool = True
    global_test_point: str = '2023-09-01 00:00:00'
    
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
        """Validate data configuration parameters."""
        if self.seq_len <= 0:
            raise ValueError("sequence_length must be positive")
        
        if self.forecast_horizon <= 0:
            raise ValueError("forecast_horizon must be positive")
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

def create_data_config(config: DictConfig):
    """Create data configuration from OmegaConf config."""
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
    