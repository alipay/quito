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

# newly added dataest should registered to this mapping using @register_dataset decorator, or it cannot be used
DATASET_MAPPING = {}

def load_datasets(
    data_config: DataConfig,
    task: TaskType,
    mode: ModeType,
    cleanup: bool = True,
    concat: bool = True,
    **kwargs
) -> Union[List[Dataset], Dataset]:
    """
    Load time series datasets from parquet files based on configuration.
    
    This function creates dataset instances for the specified task and mode,
    handling both pre-training and fine-tuning scenarios. Datasets can be
    returned as a concatenated dataset or as a list of individual datasets.
    
    Args:
        data_config (DataConfig): Configuration object containing dataset parameters
            including sequence length, forecast horizon, data directory, etc.
        task (TaskType): The task type (PRE_TRAIN, FINE_TUNE, or EVALUATE).
        mode (ModeType): The mode (TRAIN, VAL, or TEST).
        cleanup (bool, optional): Whether to cleanup/filter invalid data. Defaults to True.
        concat (bool, optional): Whether to concatenate datasets into one. 
            If False, returns list of datasets. Defaults to True.
        **kwargs: Additional keyword arguments passed to dataset constructors.
    
    Returns:
        Union[List[Dataset], Dataset]: Either a concatenated PyTorch Dataset or 
            a list of individual Dataset objects. Returns None if no datasets match criteria.
            
    Example:
        >>> data_config = DataConfig(...)
        >>> train_ds = load_datasets(data_config, TaskType.FINE_TUNE, ModeType.TRAIN)
        >>> print(f"Training samples: {len(train_ds)}")
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
                global_test_point=data_config.global_test_point,
                ds_config=ds_config,
                mode=mode,
                name=name,
                cleanup=cleanup,
                **ds_config.ds_kwargs,
            )
            logging.info(f'Loading {name} for {task} using {ds_cls.__name__} ...')
            ds_lst.append(ds)    
    
    if not ds_lst:
        return None
    
    if concat:
        ds = ConcatDataset(ds_lst)
        logging.info(f"{task} {mode} dataset size: {len(ds)} samples")
        return ds
    else:
        return ds_lst
        

def get_dataset(data_config: DataConfig, task: TaskType = TaskType.FINE_TUNE) -> Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]:
    """
    Get train, validation, and test datasets for a given task.
    
    Convenience function that loads all three dataset splits (train, validation, test)
    in a single call. Each dataset is loaded according to the task type and data configuration.
    
    Args:
        data_config (DataConfig): Data configuration object containing all dataset parameters.
        task (TaskType, optional): The task type (PRE_TRAIN, FINE_TUNE, or EVALUATE). 
            Defaults to TaskType.FINE_TUNE.
        
    Returns:
        Tuple[Optional[Dataset], Optional[Dataset], Optional[Dataset]]: A tuple containing:
            - train_ds: Training dataset (or None if not available)
            - val_ds: Validation dataset (or None if not available)
            - test_ds: Test dataset (or None if not available)
            
    Example:
        >>> data_config = DataConfig(...)
        >>> train_ds, val_ds, test_ds = get_dataset(data_config, TaskType.FINE_TUNE)
        >>> if train_ds:
        ...     print(f"Training samples: {len(train_ds)}")
    """
    train_ds = load_datasets(data_config, task, ModeType.TRAIN)
    val_ds = load_datasets(data_config, task, ModeType.VALID)
    test_ds = load_datasets(data_config, task, ModeType.TEST)
    
    return train_ds, val_ds, test_ds
    

def load_dataloader(ds: ConcatDataset, data_config: DataConfig):
    """
    Create a DataLoader from a concatenated dataset.
    
    Creates a TimeSeriesDataLoader with parameters from the data configuration,
    including batch size, shuffling, number of workers, and memory pinning.
    
    Args:
        ds (ConcatDataset): Concatenated PyTorch dataset containing time series samples.
        data_config (DataConfig): Data configuration object containing:
            - batch_size: Batch size for data loading
            - shuffle: Whether to shuffle the data
            - num_workers: Number of data loading workers
            - pin_memory: Whether to pin memory for faster GPU transfer
    
    Returns:
        TimeSeriesDataLoader: Configured data loader ready for training/evaluation.
        
    Example:
        >>> train_ds = load_datasets(data_config, TaskType.FINE_TUNE, ModeType.TRAIN)
        >>> train_loader = load_dataloader(train_ds, data_config)
        >>> for batch in train_loader:
        ...     # Process batch
    """
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
        cleanup: str = True,
        global_test_point: str = None,
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
        self.cleanup = cleanup
        self.ids = ds_config.ids
        self.global_test_point = global_test_point
        
        self._df = None
        self.data = None
        self.date_col = None
        self.time_features = None
        self.feature_cols = None
        self.mean = None
        self.std = None
        self.train_size = None
        self.valid_size = None
        self.test_size = None
        self.id_mask = None
        
        self.init_data()
        
    def _process_time_features(self, ts_series: pd.Series) -> np.ndarray:
        """
        Extract time features from datetime series.
        
        Extracts normalized time features including:
        - Hour of day (0-23 normalized to 0-1)
        - Day of week (0-6 normalized to 0-1)
        - Day of month (1-31 normalized to 0-1)
        - Month of year (1-12 normalized to 0-1)
        
        Args:
            ts_series (pd.Series): Pandas Series with datetime dtype.
        
        Returns:
            np.ndarray: Array of shape (len(ts_series), 4) containing normalized
                time features.
        """
        features = []
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
        data = self.data
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

        seq_x = data[i, s_begin:s_end, :]
        seq_y = data[i, y_begin:y_end, target_s:target_e]
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
        """
        Inverse transform normalized data back to original scale.

        If normalization was applied during initialization, this method
        reverses the normalization by multiplying by std and adding mean.

        Args:
            data (np.ndarray): Normalized data to transform back.

        Returns:
            np.ndarray: Data in original scale (if normalize=True) or
                unchanged (if normalize=False).
        """
        if self.normalize:
            return data * self.std + self.mean

        return data

    def load_data(self):
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

        self._df = df.reset_index(drop=True)

    def process_raw_df(self):
        if self._df is None:
            raise ValueError('dataframe is not loaded !!!')

        df = self._df.copy() # do not modify the original df
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
            if self.ids: # getting dataframe with only ids
                df = df[df['item_id'].isin(self.ids)]

            df_sorted = df.sort_values(['item_id', self.date_col])
            unique_ids = df_sorted['item_id'].unique()
            single_id_df = df_sorted[df_sorted['item_id'] == unique_ids[0]].reset_index(drop=True)
            n_ids = len(unique_ids)
            n_rows_per_id = len(single_id_df)
            n_cols = len(numeric_cols) - 1
            numeric_cols.remove('item_id')
            id_mask = df_sorted['item_id'].values
        else:
            df_sorted = df.sort_values([self.date_col])
            single_id_df = df_sorted.reset_index(drop=True)
            n_ids = 1
            n_rows_per_id = len(single_id_df)
            n_cols = len(numeric_cols)
            unique_ids = None
            id_mask = None

        if 'cluster' in numeric_cols:
            n_cols = n_cols - 1
            numeric_cols.remove('cluster')

        if self.global_test_point is not None:
            # if global splitting point
            train_valid_size = single_id_df[single_id_df[self.date_col] == self.global_test_point].index.item()
            train_size, valid_size, _ = self.ds_config.split(train_valid_size, test=False)
            test_size = n_rows_per_id - train_valid_size
        else:
            train_size, valid_size, test_size = self.ds_config.split(n_rows_per_id)

        data = df_sorted[numeric_cols].values
        data = data.reshape(n_ids, n_rows_per_id, n_cols) # (N_ids, n_rows, n_cols)
        if id_mask is not None:
            id_mask = id_mask.reshape(n_ids, n_rows_per_id, 1) # (N_ids, n_rows, 1), this is a mask for the ids
            id_mask = np.repeat(id_mask, n_cols, axis=-1)

        ts_series = df_sorted[self.date_col] # (n_rows, 1)
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
        if id_mask is not None:
            id_mask = id_mask[:, border_s:border_e, :]

        # # reshape the data according to features
        if self.features == Features.S:
            data = rearrange(data, 'n l c -> (n c) l 1') # (N * C) l 1
            if id_mask is not None:
                id_mask = rearrange(id_mask, 'n l c -> (n c) l 1')

        self.feature_cols = numeric_cols
        self.data = data
        self.time_features = time_features
        self.mean = mean
        self.std = std
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.id_mask = id_mask

        logging.info(f'Dataset {self.name} loaded successfully')
        logging.info(f'The splits are [{border_s}, {border_e}] for {self.mode}')
        logging.info(f'The train size is {train_size}, valid_size is {valid_size}, test_size is {test_size}')
        
    def init_data(self):
        self.load_data() # load to self._data
        self.process_raw_df() # process self._data -> self.data
        if self.cleanup: # perform clearning up on self._df to save memory
            self._df = None

    def select_user_data(self, user_id):
        if self.id_mask is None:
            raise ValueError('id_mask is not available !!!')

        assert self.id_mask.shape == self.data.shape

        mask = self.id_mask == user_id
        self.data = self.data[mask].reshape(-1, self.data.shape[1], self.data.shape[-1])
        
    def get_all_ids(self):
        if self._df is not None and 'item_id' in self._df.columns:
            return list(self._df['item_id'].unique())
        
        return None

    def get_item_cluster_mapping(self):
        if (self._df is not None) and ('item_id' in self._df.columns) and ('cluster' in self._df.columns):
            mapping = self._df[['item_id', 'cluster']].drop_duplicates()
            return mapping.set_index('item_id')['cluster'].to_dict()
        
        return {}
            
    @property
    def description(self):
        return self.ds_config


class TimeSeriesDataLoader(DataLoader):
    """
    Wrapper around PyTorch DataLoader for time series data.
    
    This is a convenience class that sets sensible defaults for time series
    data loading, including memory pinning for faster GPU transfer.
    
    Args:
        dataset (ConcatDataset): Concatenated dataset to load from.
        batch_size (int, optional): Batch size. Defaults to 32.
        shuffle (bool, optional): Whether to shuffle data. Defaults to True.
        num_workers (int, optional): Number of data loading workers.
            Defaults to 0 (single-threaded).
        pin_memory (bool, optional): Whether to pin memory for GPU transfer.
            Defaults to True.
        drop_last (bool, optional): Whether to drop last incomplete batch.
            Defaults to False.
        **kwargs: Additional arguments passed to PyTorch DataLoader.
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
        """
        Initialize the time series data loader.
        
        Args:
            dataset (ConcatDataset): Dataset to load from.
            batch_size (int): Batch size for loading.
            shuffle (bool): Whether to shuffle data.
            num_workers (int): Number of loading workers.
            pin_memory (bool): Whether to pin memory.
            drop_last (bool): Whether to drop last batch.
            **kwargs: Additional DataLoader arguments.
        """
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        ) 


