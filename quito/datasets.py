"""
Datasets for time series data in QUITO library.
"""
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from einops import rearrange

from quito.config.data import DataConfig, Freq, Features, DatasetConfig
from quito.config.training import TaskType, ModeType
from quito.utils.common import register_to_mapping
from quito.utils.data import stl_filter, naive_seasonal_decompose

# newly added dataest should registered to this mapping using @register_dataset decorator, or it cannot be used
DATASET_MAPPING = {}

def load_datasets(
    data_config: DataConfig,
    task: TaskType,
    mode: ModeType,
    **kwargs
) -> List[Dataset]:
    """
    Load time series datasets from parquets
    
    Returns:
        ConcatedDataset
    """
    ds_lst = []

    for name, ds_config in data_config.ds_config_dict.items():
        if (
            (task == TaskType.PRE_TRAIN and ds_config.is_pretrain) or
            (task != TaskType.PRE_TRAIN and not ds_config.is_pretrain)
        ):
            ds_cls = ds_config.get_ds_class(DATASET_MAPPING)
            ds = ds_cls(
                seq_len=data_config.seq_len,
                decoder_label_len=data_config.decoder_label_len,
                forecast_horizon=data_config.forecast_horizon,
                features=Features(data_config.features),
                data_dir=data_config.data_dir,
                normalize=data_config.normalize,
                ds_config=ds_config,
                mode=mode,
                name=name,
                **ds_config.ds_kwargs,
            )
            logging.info(f'Loading {name} for {task} using {ds_cls.__name__} ...')
            ds_lst.append(ds)    
    
    ds = ConcatDataset(ds_lst)
    
    logging.info(f"{task} {mode} dataset size: {len(ds)} samples")
    
    return ds
    

def load_dataloader(ds: ConcatDataset, data_config: DataConfig):
    dl = TimeSeriesDataLoader(
        dataset=ds,
        batch_size=data_config.batch_size,
        shuffle=data_config.shuffle,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
    )
    
    return dl
    
@register_to_mapping(DATASET_MAPPING)
class TimeSeriesDataset(Dataset):
    """
    Time series dataset for forecasting tasks.
    
    This dataset handles sliding window creation, normalization, and
    returns sequences in the format expected by forecasting models.
    """
    
    def __init__(
        self,
        data_dir: str,
        seq_len: int,
        decoder_label_len: int,
        forecast_horizon: int,
        features: Features,
        ds_config: DatasetConfig,
        mode: ModeType,
        normalize: bool = True,
        name: str = None,
        **kwargs
    ):
        """
        Initialize the time series dataset.
        
        Args:
            ds_config: Dataset configuration object
            mode: Mode type (train, valid, test)
        """
        # Convert to absolute path relative to project root
        data_dir_path = Path(data_dir)
        if data_dir_path.is_absolute():
            self.data_dir = data_dir_path
        else:
            # Resolve relative to project root (where quito package is)
            import quito
            project_root = Path(quito.__file__).parent.parent
            self.data_dir = (project_root / data_dir).resolve()
        self.seq_len = seq_len
        self.decoder_label_len = decoder_label_len
        self.forecast_horizon = forecast_horizon
        self.features = features
        self.normalize = normalize
        self.ds_config = ds_config
        self.target = ds_config.target
        self.name = name
        self.mode = mode
        
        self.data = None
        self.date_col = None
        self.time_features = None
        self.feature_cols = None
        self.mean = None
        self.std = None
        self.train_size = None
        self.valid_size = None
        self.test_size = None
        
        self._init_data()
        
    def _process_time_features(self, ts_series: pd.Series) -> np.ndarray:
        """Extract time features from datetime."""
        features = []
        
        if self.ds_config.freq == Freq.M:
            features.append(ts_series.dt.minute / 1440.0)
        
        # Hour of day (normalized to [0, 1])
        features.append(ts_series.dt.hour / 23.0)
        
        # Day of week (normalized to [0, 1])
        features.append(ts_series.dt.dayofweek / 6.0)
        
        # Day of month (normalized to [0, 1])
        features.append(ts_series.dt.day / 31.0)
        
        # Month of year (normalized to [0, 1])
        features.append(ts_series.dt.month / 12.0)
        
        return np.stack(features, axis=1).astype(np.float32)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        N, L, C = self.data.shape
        if self.features == Features.S:
            # Univariate to univariate
            return (L - self.seq_len - self.forecast_horizon + 1) * N * C
        else:
            # Multivariate to univariate
            return (L - self.seq_len - self.forecast_horizon + 1) * N
    
    def _fetch_sample_idx(self, idx):
        """
        Fetch the sample index from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (i, j) indices
        """
        N, L, C = self.data.shape
        len_per_ts = L - self.seq_len - self.forecast_horizon + 1
        i = idx // len_per_ts
        j = idx % len_per_ts
        
        return i, j
         
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Tuple of (seq_x, seq_y, seq_x_mark, seq_y_mark) tensors
            - seq_x: Input sequence [seq_len, n_features]
            - seq_y: Output sequence [label_len + pred_len, n_features]
            - seq_x_mark: Time features for input [seq_len, time_features]
            - seq_y_mark: Time features for output [label_len + pred_len, time_features]
        """
        i, j = self._fetch_sample_idx(idx)
        # get_target
        if self.features == Features.M:
            target_s = 0
            target_e = self.data.shape[-1]
        else:
            target_s = 0
            target_e = 1
            
        # Input sequence
        s_begin = j
        s_end = s_begin + self.seq_len
        y_begin = s_end - self.decoder_label_len
        y_end = s_end + self.forecast_horizon 

        seq_x = self.data[i, s_begin:s_end, :]
        seq_y = self.data[i, y_begin:y_end, target_s:target_e]
        seq_x_mark = self.time_features[s_begin:s_end, :]
        seq_y_mark = self.time_features[y_begin:y_end, :]
        
        # Convert to torch tensors
        seq_x = torch.from_numpy(seq_x).float()
        seq_y = torch.from_numpy(seq_y).float()
        seq_x_mark = torch.from_numpy(seq_x_mark).float()
        seq_y_mark = torch.from_numpy(seq_y_mark).float()
        
        assert seq_x.ndim == 2
        assert seq_y.ndim == 2
        assert seq_x_mark.ndim == 2
        assert seq_y_mark.ndim == 2

        out_dict = {
            'x': seq_x, 
            'y': seq_y,
            'x_mark': seq_x_mark,
            'y_mark': seq_y_mark
            }
        
        return out_dict
    
    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Inverse transform scaled data back to original scale."""
        if self.normalize:
            return data * self.std + self.mean
        
        return data
    
    def _init_data(self):
        # Process dataframe - use absolute path
        file_path = (self.data_dir / self.ds_config.file_name).resolve()
        df = pd.read_parquet(str(file_path))
        # Identify columns
        for col in ['date_time', 'date', 'datetime', 'timestamp']:
            if col in df.columns:
                self.date_col = col
                break
        else:
            raise ValueError("No date column found in dataset")
        
        # datetime col to pd.datetime
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        # Move target to first place if exists
        if self.target:
            columns = [self.target] + [c for c in df.columns if c != self.target]
            df = df[columns]
        
        # Extract numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove item_id if present (used for grouping, not features)
        if 'item_id' in numeric_cols:
            # make the data 3D of shape (n_item_id, n_rows, n_cols), assume each item_id has same length
            df_sorted = df.sort_values(['item_id', self.date_col])
            unique_ids = df_sorted['item_id'].unique()
            n_ids = len(unique_ids)
            n_rows_per_id = df_sorted.groupby('item_id').size().iloc[0]
            n_cols = len(numeric_cols) - 1
            numeric_cols.remove('item_id')
        else:
            df_sorted = df.sort_values([self.date_col])
            n_ids = 1
            n_rows_per_id = len(df_sorted)
            n_cols = len(numeric_cols)
            unique_ids = None
        
        # apply preprocessing to each time series
        # logging.info('Performing anomaly detection and imputation ...')
        # for c in numeric_cols:
        #     if unique_ids is not None:
        #         for i in unique_ids:
        #             sub_df = df_sorted.loc[df['item_id'] == i, c]
        #             cleaned_array = self.remove_anomaly(sub_df.values)
        #             df_sorted.loc[df['item_id'] == i, c] = cleaned_array
        #     else:
        #         sub_df = df_sorted.loc[:, c]
        #         cleaned_array = self.remove_anomaly(sub_df.values)
        #         df_sorted.loc[:, c] = cleaned_array
        
        # na impurtation
        # df_sorted[numeric_cols] = self.impute_na(df_sorted[numeric_cols])
        data = df_sorted[numeric_cols].values
        data = data.reshape(n_ids, n_rows_per_id, n_cols) # (N_ids, n_rows, n_cols)
        ts_series = df_sorted[self.date_col] # (n_rows, 1)
        
        # calulate split and scale data
        train_size, valid_size, test_size = self.ds_config.split(data.shape[1])
        mean = np.mean(data[:, :train_size, :], axis=1, keepdims=True)
        std = np.std(data[:, :train_size, :], axis=1, keepdims=True) + 1e-8
        data = (data - mean) / std
        # Time features
        time_features = self._process_time_features(ts_series) # L * n_ids, num_time_features
        time_features = time_features.reshape(n_ids, n_rows_per_id, -1) 
        # select the appropriate data according to mode
        if self.mode == ModeType.TRAIN:
            border_s, border_e = 0, train_size
        elif self.mode == ModeType.VALID:
            border_s, border_e = train_size - self.seq_len, train_size + valid_size
        else:
            border_s, border_e = train_size + valid_size - self.seq_len, train_size + valid_size + test_size
        
        data = data[:, border_s:border_e, :]
        time_features = time_features[0, border_s:border_e, :] # just use the first time features, because other ids have same time features
        
        # reshape the data according to features
        if self.features == Features.S:
            data = rearrange(data, 'n l c -> (n c) l 1') # (N * C) l 1

        self.feature_cols = numeric_cols
        self.data = data
        self.time_features = time_features
        self.mean = mean
        self.std = std
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

        logging.info(f'Dataset {self.name} loaded successfully')
        logging.info(f'The splits are [{border_s}, {border_e}] for {self.mode}')
    
    def remove_anomaly(self, data: np.array) -> np.array:
        """Remove anomaly from the data."""
        # The input data is a pd.Series
        anomaly_preprocess_method = self.ds_config.anomaly_remove_method
        if anomaly_preprocess_method == 'stl':
            data = stl_filter(data, self.ds_config.freq)
        elif anomaly_preprocess_method == 'naive_seasonal_decompose':
            data = naive_seasonal_decompose(data, self.ds_config.freq)
        
        return data
    
    def impute_na(self, data: pd.DataFrame):
        """Impute NA values in the data."""
        impute_method = self.ds_config.na_impute_method
        if impute_method == 'linear':
            data = data.interpolate(method='linear', axis=1)
        
        if data.isna().any().sum():
            logging.warning(f'NA values found in data !!!')

        return data

    @property
    def description(self):
        return self.ds_config
    
    
class TimeSeriesDataLoader(DataLoader):
    """
    Wrapper around PyTorch DataLoader for time series data.
    
    This is a convenience class that sets sensible defaults for time series.
    """
    
    def __init__(
        self,
        dataset: ConcatDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        **kwargs
    ):
        """Initialize the data loader."""
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        ) 

