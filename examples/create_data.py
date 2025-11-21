#!/usr/bin/env python3
"""
Create Dummy Time Series Data

Generates sample Parquet files for testing the training pipeline.
Self-contained with no dependencies on other example scripts.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def create_dummy_data(output_dir="examples/datasets/parquet_data/open_hour_train", 
                      num_files=1, 
                      num_rows=1000):
    """
    Create dummy time series data in Parquet format.
    
    Args:
        output_dir: Directory to save Parquet files
        num_files: Number of files to create
        num_rows: Number of rows per file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Creating Dummy Time Series Data")
    print("=" * 60)
    
    for i in range(1, num_files + 1):
        print(f"\nGenerating file {i}/{num_files}...")
        
        # Generate timestamps (hourly data)
        dates = pd.date_range(start='2023-01-01', periods=num_rows, freq='h')
        
        # Generate synthetic time series data
        # Trend + Seasonality + Noise
        trend = np.linspace(0, 10, num_rows)
        seasonality = 5 * np.sin(2 * np.pi * np.arange(num_rows) / 24)  # Daily pattern
        noise = np.random.randn(num_rows) * 0.5
        value = trend + seasonality + noise
        
        # Additional features
        feature1 = np.random.randn(num_rows)
        feature2 = np.random.randn(num_rows)
        
        # Create DataFrame
        df = pd.DataFrame({
            'date': dates,
            'value': value,
            'feature1': feature1,
            'feature2': feature2,
            'item_id': 0  # Single time series
        })
        
        # Save as Parquet
        filename = f"hour_train_hour_p{i}.parquet"
        filepath = output_path / filename
        df.to_parquet(filepath, index=False, engine='pyarrow')
        
        print(f"✅ Created: {filepath}")
        print(f"   Shape: {df.shape}")
        print(f"   Columns: {list(df.columns)}")
    
    print("\n" + "=" * 60)
    print(f"✅ Successfully created {num_files} Parquet file(s)")
    print(f"📁 Location: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    # Create 1 dummy file with 1000 rows for quick testing
    create_dummy_data(num_files=1, num_rows=1000)
    
    print("\n💡 Usage:")
    print("   python examples/train_chronos.py")
    print("   python examples/train_moriai.py")
    print("   python examples/train_patchtst.py")

