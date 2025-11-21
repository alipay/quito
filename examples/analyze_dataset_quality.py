#!/usr/bin/env python3
"""
Dataset Quality Analysis Example

This script demonstrates how to evaluate time series dataset quality using QUITO's
dataset quality toolkit.

Features:
- Forecastability (spectral entropy-based)
- Seasonality strength (STL decomposition)
- Missing data ratio
- Effective length
- Coefficient of variation
- ADF stationarity test (optional)

Usage:
    python examples/analyze_dataset_quality.py

Dependencies:
    Core: numpy, scipy
    Optional: statsmodels (for seasonality), arch (for ADF test), matplotlib (for plots)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import numpy as np
import pandas as pd

from quito.utils.dataset_quality import (
    evaluate_series,
    evaluate_dataset,
    print_dataset_report
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_parquet_as_series_list(parquet_path: str, value_col: str = 'value') -> list:
    """
    Load a parquet file and extract time series.
    Assumes either:
    - Single series (one value column)
    - Multiple series (item_id column with different IDs)
    """
    df = pd.read_parquet(parquet_path)
    
    # Check if multiple series (has item_id or similar)
    if 'item_id' in df.columns:
        series_list = []
        for item_id in df['item_id'].unique():
            item_data = df[df['item_id'] == item_id][value_col].values
            series_list.append(item_data)
        return series_list
    else:
        # Single series
        return [df[value_col].values]


def example_single_series():
    """Example 1: Evaluate a single time series."""
    logger.info("="*80)
    logger.info("Example 1: Single Series Evaluation")
    logger.info("="*80)
    
    # Generate synthetic data with trend and seasonality
    np.random.seed(42)
    t = np.arange(1000)
    trend = 0.01 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 24)  # Daily pattern
    noise = np.random.randn(1000) * 2
    series = trend + seasonal + noise
    
    # Add some missing values
    missing_idx = np.random.choice(1000, size=50, replace=False)
    series[missing_idx] = np.nan
    
    # Evaluate
    result = evaluate_series(
        series,
        period=24,  # Daily seasonality
        compute_adf=True,  # Compute stationarity test
        adf_fill='mean'
    )
    
    logger.info(f"\nSeries Quality Metrics:")
    logger.info(f"  Forecastability:    {result.forecastability:.4f}")
    logger.info(f"  Seasonality Strength: {result.season_strength:.4f}")
    logger.info(f"  Missing Ratio:      {result.missing_ratio:.4f} ({result.missing_ratio*100:.2f}%)")
    logger.info(f"  Effective Length:   {result.eff_length}")
    logger.info(f"  Coefficient of Var: {result.cv:.4f}")
    logger.info(f"  ADF Statistic:      {result.adf_stat:.4f}")
    
    # Interpretation
    if result.forecastability > 0.7:
        logger.info("\n  ✓ High forecastability - good for forecasting!")
    elif result.forecastability > 0.5:
        logger.info("\n  ~ Moderate forecastability")
    else:
        logger.info("\n  ✗ Low forecastability - may be difficult to forecast")


def example_dataset_evaluation():
    """Example 2: Evaluate an entire dataset."""
    logger.info("\n" + "="*80)
    logger.info("Example 2: Dataset Evaluation")
    logger.info("="*80)
    
    # Generate multiple series
    np.random.seed(123)
    series_list = []
    
    for i in range(50):
        length = np.random.randint(500, 1500)
        t = np.arange(length)
        
        # Different characteristics for each series
        trend = 0.005 * t * (1 + 0.5 * np.random.randn())
        seasonal = np.random.uniform(5, 15) * np.sin(2 * np.pi * t / 24)
        noise = np.random.randn(length) * np.random.uniform(1, 5)
        s = trend + seasonal + noise + np.random.uniform(0, 100)
        
        # Random missing data
        if np.random.rand() > 0.5:
            n_missing = np.random.randint(0, int(0.1 * length))
            missing_idx = np.random.choice(length, size=n_missing, replace=False)
            s[missing_idx] = np.nan
        
        series_list.append(s)
    
    # Evaluate dataset
    results = evaluate_dataset(
        series_list,
        period=24,
        compute_adf=False,  # Skip ADF for speed
        verbose=True
    )
    
    # Print formatted report
    print_dataset_report(results)


def example_load_real_data():
    """Example 3: Load and evaluate real parquet data."""
    logger.info("\n" + "="*80)
    logger.info("Example 3: Real Data Evaluation")
    logger.info("="*80)
    
    # Path to example data
    data_path = Path(__file__).parent / "datasets" / "parquet_data" / "open_hour_train" / "hour_train_hour_p1.parquet"
    
    if not data_path.exists():
        logger.warning(f"Data file not found: {data_path}")
        logger.info("Run 'python examples/create_data.py' to generate sample data first.")
        return
    
    # Load data
    logger.info(f"Loading data from {data_path}...")
    series_list = load_parquet_as_series_list(str(data_path))
    logger.info(f"Loaded {len(series_list)} time series")
    
    if len(series_list) == 0:
        logger.error("No series found in data!")
        return
    
    # Evaluate
    results = evaluate_dataset(
        series_list,
        period=24,  # Hourly data with daily pattern
        compute_adf=False,
        verbose=True
    )
    
    # Print report
    print_dataset_report(results)


def main():
    """Run all examples."""
    
    # Example 1: Single series
    example_single_series()
    
    # Example 2: Multiple series (dataset)
    example_dataset_evaluation()
    
    # Example 3: Real data from parquet
    example_load_real_data()
    
    logger.info("\n" + "="*80)
    logger.info("Dataset Quality Analysis Examples Completed!")
    logger.info("="*80)
    logger.info("\nNext steps:")
    logger.info("  1. Analyze your own data with evaluate_dataset()")
    logger.info("  2. Compare multiple datasets with compare_datasets()")
    logger.info("  3. See examples/compare_datasets_quality.py for cross-dataset comparison")


if __name__ == "__main__":
    main()

