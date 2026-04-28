"""
Data utilities for QUITO library.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def create_data_directory_structure(base_path: Union[str, Path] = "./data") -> Path:
    """
    Create a standard data directory structure for QuitoBench projects.
    
    Sets up a well-organized directory hierarchy for storing raw data, processed
    data, train/val/test splits, synthetic datasets, benchmarks, cache, and exports.
    Also creates a README.md file explaining the structure.
    
    Args:
        base_path (Union[str, Path], optional): Base path for the data directory.
            Defaults to "./data".
        
    Returns:
        Path: Path object pointing to the created data directory.
        
    Example:
        >>> data_dir = create_data_directory_structure("./my_data")
        >>> # Creates: my_data/raw/, my_data/processed/, my_data/train/, etc.
    """
    base_path = Path(base_path)
    
    # Create main data directory
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "raw",           # Raw, unprocessed data
        "processed",     # Processed and cleaned data
        "train",         # Training data
        "val",           # Validation data
        "test",          # Test data
        "synthetic",     # Synthetic datasets
        "benchmarks",    # Benchmark datasets (UCR, UEA, etc.)
        "cache",         # Cached preprocessed data
        "exports",       # Exported results and predictions
    ]
    
    for subdir in subdirs:
        (base_path / subdir).mkdir(exist_ok=True)
    
    # Create a README file explaining the structure
    readme_content = """# QUITO Data Directory Structure

This directory contains all data files for your QUITO project.

## Directory Structure:

- `raw/`: Raw, unprocessed data files
- `processed/`: Processed and cleaned data files
- `train/`: Training datasets
- `val/`: Validation datasets  
- `test/`: Test datasets
- `synthetic/`: Synthetic datasets generated for training/testing
- `benchmarks/`: Standard benchmark datasets (UCR, UEA, etc.)
- `cache/`: Cached preprocessed data for faster loading
- `exports/`: Exported results, predictions, and model outputs

## File Naming Convention:

- Use descriptive names with underscores: `stock_prices_2023.csv`
- Include date/version information when relevant
- Use consistent file extensions (.csv, .json, .npz, .pkl)

## Data Format Guidelines:

- CSV files should have clear column headers
- Time series data should be in chronological order
- Missing values should be handled appropriately
- Include metadata files (.json) for complex datasets
"""
    
    with open(base_path / "README.md", "w") as f:
        f.write(readme_content)
    
    print(f"Created data directory structure at: {base_path}")
    return base_path


def get_default_data_path() -> Path:
    """
    Get the default data directory path.
    
    Returns:
        Path: Path object pointing to the default data directory (./data).
    """
    return Path("./data")


def save_dataset(data: Dict[str, Any], filename: str, data_dir: Optional[Union[str, Path]] = None, 
                 format: str = "npz") -> Path:
    """
    Save a dataset dictionary to disk in various formats.
    
    Supports multiple file formats (NPZ, PKL, JSON) and automatically handles
    PyTorch tensor to NumPy array conversion. Creates parent directories if needed.
    
    Args:
        data (Dict[str, Any]): Dataset dictionary containing arrays, tensors, or
            other serializable data structures.
        filename (str): Name of the file without extension (e.g., "my_dataset").
        data_dir (Optional[Union[str, Path]], optional): Directory to save the file.
            Defaults to ./data/processed if None.
        format (str, optional): File format. Options: 'npz', 'pkl', 'json'.
            Defaults to "npz".
    
    Returns:
        Path: Path object pointing to the saved file.
        
    Raises:
        ValueError: If an unsupported format is specified.
        
    Example:
        >>> dataset = {'train': np.array([1, 2, 3]), 'test': np.array([4, 5, 6])}
        >>> path = save_dataset(dataset, "my_data", format="npz")
    """
    if data_dir is None:
        data_dir = get_default_data_path() / "processed"
    else:
        data_dir = Path(data_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if format == "npz":
        filepath = data_dir / f"{filename}.npz"
        # Convert tensors to numpy arrays
        numpy_data = {}
        for key, value in data.items():
            if torch.is_tensor(value):
                numpy_data[key] = value.cpu().numpy()
            else:
                numpy_data[key] = value
        np.savez(filepath, **numpy_data)
    
    elif format == "pkl":
        filepath = data_dir / f"{filename}.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
    
    elif format == "json":
        filepath = data_dir / f"{filename}.json"
        # Convert numpy arrays and tensors to lists
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            elif torch.is_tensor(value):
                json_data[key] = value.cpu().numpy().tolist()
            else:
                json_data[key] = value
        
        with open(filepath, "w") as f:
            json.dump(json_data, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    print(f"Saved dataset to: {filepath}")
    return filepath


def load_data_from_file(filename: str, data_dir: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load a dataset from disk, automatically detecting file format.
    
    Attempts to load a dataset file, trying multiple extensions (.npz, .pkl, .json, .csv)
    if the exact filename is not found. Supports NumPy, pickle, JSON, and CSV formats.
    
    Args:
        filename (str): Name of the file with or without extension.
        data_dir (Optional[Union[str, Path]], optional): Directory to load from.
            Defaults to ./data/processed if None.
    
    Returns:
        Dict[str, Any]: Dataset dictionary containing the loaded data.
        
    Raises:
        FileNotFoundError: If the dataset file cannot be found.
        ValueError: If the file format is unsupported.
        
    Example:
        >>> data = load_data_from_file("my_dataset", data_dir="./data/processed")
        >>> # Automatically tries: my_dataset.npz, my_dataset.pkl, etc.
    """
    if data_dir is None:
        data_dir = get_default_data_path() / "processed"
    else:
        data_dir = Path(data_dir)
    
    # Try different extensions if not provided
    filepath = data_dir / filename
    if not filepath.exists():
        for ext in [".npz", ".pkl", ".json", ".csv"]:
            test_path = data_dir / f"{filename}{ext}"
            if test_path.exists():
                filepath = test_path
                break
    
    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")
    
    if filepath.suffix == ".npz":
        data = dict(np.load(filepath))
    elif filepath.suffix == ".pkl":
        with open(filepath, "rb") as f:
            data = pickle.load(f)
    elif filepath.suffix == ".json":
        with open(filepath, "r") as f:
            data = json.load(f)
    elif filepath.suffix == ".csv":
        df = pd.read_csv(filepath)
        data = {"data": df.values, "columns": df.columns.tolist()}
    else:
        raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    return data


def normalize_data(data: np.ndarray, method: str = "standard", 
                   scaler: Optional[Any] = None) -> Tuple[np.ndarray, Any]:
    """
    Normalize time series data using various scaling methods.
    
    Applies normalization to time series data, either fitting a new scaler or
    using a pre-fitted one. Supports standard scaling (zero mean, unit variance)
    and min-max scaling (0-1 range).
    
    Args:
        data (np.ndarray): Input time series data array (1D or 2D).
        method (str, optional): Normalization method. Options:
            - 'standard': Zero mean, unit variance (StandardScaler)
            - 'minmax': Scale to [0, 1] range (MinMaxScaler)
            Defaults to "standard".
        scaler (Optional[Any], optional): Pre-fitted scaler object. If provided,
            uses it for transformation instead of fitting a new one. Defaults to None.
    
    Returns:
        Tuple[np.ndarray, Any]: A tuple containing:
            - normalized_data: Normalized data array
            - scaler: Fitted scaler object (can be used for denormalization)
            
    Raises:
        ValueError: If an unknown normalization method is specified.
        
    Example:
        >>> data = np.array([1, 2, 3, 4, 5])
        >>> normalized, scaler = normalize_data(data, method="standard")
        >>> # Use scaler later for denormalization
    """
    if scaler is None:
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization method: {method}")
        
        # Fit and transform
        if data.ndim == 1:
            data_reshaped = data.reshape(-1, 1)
            normalized = scaler.fit_transform(data_reshaped).flatten()
        else:
            normalized = scaler.fit_transform(data)
    else:
        # Just transform
        if data.ndim == 1:
            data_reshaped = data.reshape(-1, 1)
            normalized = scaler.transform(data_reshaped).flatten()
        else:
            normalized = scaler.transform(data)
    
    return normalized, scaler


def denormalize_data(data: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Denormalize time series data using a fitted scaler.
    
    Reverses the normalization transformation applied by normalize_data(),
    restoring the data to its original scale.
    
    Args:
        data (np.ndarray): Normalized data array (1D or 2D).
        scaler (Any): Fitted scaler object from normalize_data().
    
    Returns:
        np.ndarray: Denormalized data array in original scale.
        
    Example:
        >>> normalized, scaler = normalize_data(original_data)
        >>> restored = denormalize_data(normalized, scaler)
        >>> np.allclose(original_data, restored)  # Should be True
    """
    if data.ndim == 1:
        data_reshaped = data.reshape(-1, 1)
        denormalized = scaler.inverse_transform(data_reshaped).flatten()
    else:
        denormalized = scaler.inverse_transform(data)
    
    return denormalized


def split_sequences(data: np.ndarray, sequence_length: int, 
                    forecast_horizon: int = 1, step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split time series data into input-output sequence pairs for training.
    
    Creates sliding windows of input sequences and corresponding target sequences
    for supervised learning. Useful for preparing time series data for RNN/Transformer models.
    
    Args:
        data (np.ndarray): Input time series data (1D array).
        sequence_length (int): Length of input sequences (lookback window).
        forecast_horizon (int, optional): Number of future steps to predict.
            Defaults to 1.
        step (int, optional): Step size between sequences. Use 1 for overlapping
            windows, larger values for non-overlapping. Defaults to 1.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - input_sequences: Array of shape (n_samples, sequence_length)
            - target_sequences: Array of shape (n_samples, forecast_horizon)
            
    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        >>> X, y = split_sequences(data, sequence_length=3, forecast_horizon=2)
        >>> # X: [[1,2,3], [2,3,4], ...], y: [[4,5], [5,6], ...]
    """
    X, y = [], []
    
    for i in range(0, len(data) - sequence_length - forecast_horizon + 1, step):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])
    
    return np.array(X), np.array(y)


def create_sliding_windows(data: np.ndarray, window_size: int, 
                          step_size: int = 1) -> np.ndarray:
    """
    Create sliding windows from time series data.
    
    Generates overlapping or non-overlapping windows of fixed size from a time series.
    Useful for creating training samples or analyzing local patterns.
    
    Args:
        data (np.ndarray): Input time series data (1D array).
        window_size (int): Size of each sliding window.
        step_size (int, optional): Step size between windows. Use 1 for overlapping
            windows, window_size for non-overlapping. Defaults to 1.
    
    Returns:
        np.ndarray: Array of sliding windows with shape (n_windows, window_size).
        
    Example:
        >>> data = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> windows = create_sliding_windows(data, window_size=3, step_size=2)
        >>> # Result: [[1,2,3], [3,4,5], [5,6,7]]
    """
    windows = []
    
    for i in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[i:i + window_size])
    
    return np.array(windows)


def pad_sequences(sequences: List[np.ndarray], max_length: Optional[int] = None,
                  padding: str = "post", value: float = 0.0) -> np.ndarray:
    """
    Pad sequences to the same length for batch processing.
    
    Pads or truncates sequences to a uniform length, which is required for
    efficient batch processing in neural networks. Supports pre-padding and
    post-padding strategies.
    
    Args:
        sequences (List[np.ndarray]): List of variable-length sequences to pad.
        max_length (Optional[int], optional): Target length for all sequences.
            If None, uses the length of the longest sequence. Defaults to None.
        padding (str, optional): Padding strategy. Options:
            - 'post': Add padding at the end of sequences
            - 'pre': Add padding at the beginning of sequences
            Defaults to "post".
        value (float, optional): Value to use for padding. Defaults to 0.0.
    
    Returns:
        np.ndarray: Padded sequences array with shape (n_sequences, max_length).
        
    Example:
        >>> seqs = [np.array([1, 2]), np.array([1, 2, 3, 4])]
        >>> padded = pad_sequences(seqs, max_length=5, padding="post")
        >>> # Result: [[1, 2, 0, 0, 0], [1, 2, 3, 4, 0]]
    """
    if max_length is None:
        max_length = max(len(seq) for seq in sequences)
    
    padded = []
    for seq in sequences:
        if len(seq) >= max_length:
            padded.append(seq[:max_length])
        else:
            pad_length = max_length - len(seq)
            if padding == "post":
                padded_seq = np.concatenate([seq, np.full(pad_length, value)])
            else:  # pre
                padded_seq = np.concatenate([np.full(pad_length, value), seq])
            padded.append(padded_seq)
    
    return np.array(padded)


def split_train_val_test(data: np.ndarray, train_ratio: float = 0.7, 
                        val_ratio: float = 0.15, test_ratio: float = 0.15,
                        random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets with random shuffling.
    
    Randomly shuffles and splits data into three subsets according to specified
    ratios. Useful for time series data where temporal ordering is not critical
    or for cross-validation setups.
    
    Args:
        data (np.ndarray): Input data array to split.
        train_ratio (float, optional): Proportion of data for training.
            Defaults to 0.7.
        val_ratio (float, optional): Proportion of data for validation.
            Defaults to 0.15.
        test_ratio (float, optional): Proportion of data for testing.
            Defaults to 0.15.
        random_state (Optional[int], optional): Random seed for reproducibility.
            If None, uses current random state. Defaults to None.
    
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - train_data: Training subset
            - val_data: Validation subset
            - test_data: Test subset
            
    Raises:
        ValueError: If ratios do not sum to 1.0.
        
    Example:
        >>> data = np.arange(100)
        >>> train, val, test = split_train_val_test(data, 0.7, 0.15, 0.15, random_state=42)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(data)
    indices = np.random.permutation(n_samples)
    
    train_size = int(train_ratio * n_samples)
    val_size = int(val_ratio * n_samples)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    return data[train_indices], data[val_indices], data[test_indices]


def save_data_splits(train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray,
                     dataset_name: str, data_dir: Optional[Union[str, Path]] = None):
    """
    Save train/val/test splits to separate directories.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        dataset_name: Name of the dataset
        data_dir: Base data directory
    """
    if data_dir is None:
        data_dir = get_default_data_path()
    else:
        data_dir = Path(data_dir)
    
    # Save to respective directories
    np.save(data_dir / "train" / f"{dataset_name}.npy", train_data)
    np.save(data_dir / "val" / f"{dataset_name}.npy", val_data)
    np.save(data_dir / "test" / f"{dataset_name}.npy", test_data)
    
    # Save metadata
    metadata = {
        "dataset_name": dataset_name,
        "train_shape": train_data.shape,
        "val_shape": val_data.shape,
        "test_shape": test_data.shape,
        "created_at": pd.Timestamp.now().isoformat(),
    }
    
    with open(data_dir / "processed" / f"{dataset_name}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Saved data splits for {dataset_name}:")
    print(f"  Train: {train_data.shape}")
    print(f"  Val: {val_data.shape}")
    print(f"  Test: {test_data.shape}")


def load_data_splits(dataset_name: str, data_dir: Optional[Union[str, Path]] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/val/test splits from directories.
    
    Args:
        dataset_name: Name of the dataset
        data_dir: Base data directory
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if data_dir is None:
        data_dir = get_default_data_path()
    else:
        data_dir = Path(data_dir)
    
    train_data = np.load(data_dir / "train" / f"{dataset_name}.npy")
    val_data = np.load(data_dir / "val" / f"{dataset_name}.npy")
    test_data = np.load(data_dir / "test" / f"{dataset_name}.npy")
    
    return train_data, val_data, test_data


def generate_synthetic_dataset(dataset_type: str = "sine", num_samples: int = 1000,
                              sequence_length: int = 100, num_features: int = 1,
                              noise_level: float = 0.1, save_path: Optional[str] = None) -> Dict[str, np.ndarray]:
    """
    Generate synthetic time series datasets for testing and experimentation.
    
    Creates artificial time series data with various patterns (sine waves, trends,
    seasonality, random walks) for algorithm testing, benchmarking, and debugging.
    
    Args:
        dataset_type (str, optional): Type of synthetic data pattern. Options:
            - 'sine': Sinusoidal wave pattern
            - 'trend': Linear trend with noise
            - 'seasonal': Combined seasonal and trend components
            - 'random_walk': Random walk process
            Defaults to "sine".
        num_samples (int, optional): Number of independent time series samples
            to generate. Defaults to 1000.
        sequence_length (int, optional): Length of each time series sequence.
            Defaults to 100.
        num_features (int, optional): Number of features/dimensions per sample.
            Defaults to 1 (univariate).
        noise_level (float, optional): Standard deviation of Gaussian noise
            to add. Defaults to 0.1.
        save_path (Optional[str], optional): Path to save the generated dataset.
            If None, dataset is not saved. Defaults to None.
    
    Returns:
        Dict[str, np.ndarray]: Dictionary containing:
            - 'data': Generated dataset array
            - 'dataset_type': Type of dataset generated
            - 'num_samples': Number of samples
            - 'sequence_length': Length of sequences
            - 'num_features': Number of features
            - 'noise_level': Noise level used
            
    Raises:
        ValueError: If an unknown dataset_type is specified.
        
    Example:
        >>> dataset = generate_synthetic_dataset("sine", num_samples=100, sequence_length=50)
        >>> print(dataset['data'].shape)  # (100, 50)
    """
    np.random.seed(42)  # For reproducibility
    
    if dataset_type == "sine":
        t = np.linspace(0, 4 * np.pi, sequence_length)
        data = np.sin(t) + np.random.normal(0, noise_level, sequence_length)
    
    elif dataset_type == "trend":
        t = np.linspace(0, 1, sequence_length)
        data = t + np.random.normal(0, noise_level, sequence_length)
    
    elif dataset_type == "seasonal":
        t = np.linspace(0, 4 * np.pi, sequence_length)
        seasonal = np.sin(t) + 0.5 * np.sin(2 * t)
        trend = 0.01 * np.arange(sequence_length)
        data = seasonal + trend + np.random.normal(0, noise_level, sequence_length)
    
    elif dataset_type == "random_walk":
        data = np.cumsum(np.random.normal(0, 1, sequence_length))
        data = data + np.random.normal(0, noise_level, sequence_length)
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Replicate for multiple samples and features
    if num_features > 1:
        dataset = np.random.rand(num_samples, sequence_length, num_features)
        for i in range(num_features):
            for j in range(num_samples):
                dataset[j, :, i] = data + np.random.normal(0, noise_level * 0.1, sequence_length)
    else:
        dataset = np.tile(data, (num_samples, 1))
        # Add some variation between samples
        for i in range(num_samples):
            dataset[i] += np.random.normal(0, noise_level * 0.1, sequence_length)
    
    result = {
        "data": dataset,
        "dataset_type": dataset_type,
        "num_samples": num_samples,
        "sequence_length": sequence_length,
        "num_features": num_features,
        "noise_level": noise_level,
    }
    
    if save_path:
        save_dataset(result, save_path, format="npz")

    return result
