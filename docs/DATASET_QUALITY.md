# Dataset Quality Analysis

QUITO includes a comprehensive toolkit for evaluating the quality of time series datasets. This helps you understand your data, compare different datasets, and make informed decisions about which data to use for training and evaluation.

## Overview

The dataset quality toolkit provides:
- **Per-series metrics**: Evaluate individual time series
- **Dataset-level summaries**: Aggregate statistics across multiple series
- **Cross-dataset comparison**: Rank datasets by quality using composite scores
- **Visualization**: Plot quality distributions and comparisons

## Core Metrics

### 1. Forecastability (0-1)
**Definition**: `1 - spectral_entropy`

Measures how predictable a time series is based on its frequency spectrum.
- **1.0**: Perfect predictability (strong patterns)
- **0.5**: Moderate predictability
- **0.0**: Random noise (unpredictable)

**Interpretation**:
- `> 0.7`: Excellent for forecasting
- `0.5-0.7`: Good for forecasting
- `< 0.5`: Challenging for forecasting

### 2. Seasonality Strength (0-1)
**Definition**: `1 - Var(residuals) / Var(seasonal + residuals)`

Measures the strength of seasonal patterns using STL decomposition.
- **1.0**: Pure seasonality
- **0.5**: Moderate seasonality
- **0.0**: No seasonality

**Requirements**: `statsmodels` library

### 3. Missing Ratio (0-1)
**Definition**: Fraction of NaN values in the series

- **0.0**: No missing data
- **< 0.05**: Acceptable
- **> 0.15**: Poor data quality

### 4. Effective Length
**Definition**: Count of non-NaN values

Longer series generally provide more information for training.

### 5. Coefficient of Variation (CV)
**Definition**: `std / |mean|`

Measures relative variability. High CV indicates high volatility.

### 6. ADF Statistic
**Definition**: Augmented Dickey-Fuller test for stationarity

- **< -3.43**: Strong stationarity (1% significance)
- **< -2.86**: Moderate stationarity (5% significance)
- **> -2.86**: Non-stationary

**Requirements**: `arch` library

## Composite Quality Score

For cross-dataset comparison, QUITO computes a weighted composite score:

```
QualityScore = 0.45 × Forecastability 
             + 0.25 × Seasonality
             + 0.15 × (1 - MissingRatio)
             + 0.15 × LengthNorm
```

Where:
- **Forecastability (45%)**: Most important for forecasting success
- **Seasonality (25%)**: Indicates presence of learnable patterns
- **Completeness (15%)**: Penalizes missing data
- **Length (15%)**: Rewards longer series (log-normalized)

**Score Ranges**:
- **0.8-1.0**: Excellent quality
- **0.6-0.8**: Good quality
- **0.4-0.6**: Fair quality
- **0.0-0.4**: Poor quality

## Installation

### Core Features
Core functionality (forecastability, missing ratio, CV) works with base dependencies:
```bash
# Already included in requirements.txt
pip install numpy scipy
```

### Full Features
For seasonality strength, ADF test, and plotting:
```bash
pip install statsmodels arch matplotlib
```

## Usage Examples

### Example 1: Evaluate a Single Series

```python
from quito.utils.dataset_quality import evaluate_series
import numpy as np

# Your time series data
series = np.random.randn(1000)

# Evaluate
result = evaluate_series(
    series,
    period=24,        # For hourly data with daily patterns
    compute_adf=True  # Include stationarity test
)

print(f"Forecastability: {result.forecastability:.4f}")
print(f"Seasonality: {result.season_strength:.4f}")
print(f"Missing: {result.missing_ratio:.4f}")
print(f"Length: {result.eff_length}")
```

### Example 2: Evaluate a Dataset

```python
from quito.utils.dataset_quality import evaluate_dataset, print_dataset_report

# List of time series
series_list = [np.random.randn(1000) for _ in range(50)]

# Evaluate
results = evaluate_dataset(
    series_list,
    period=24,
    compute_adf=False,  # Skip for speed
    verbose=True
)

# Print formatted report
print_dataset_report(results)
```

Output:
```
======================================================================
QUITO: DATASET QUALITY EVALUATION REPORT
======================================================================

Dataset Statistics:
  Total Time Points: 50,000
  Number of Series:  50
  Average Length:    1,000.0

Weighted Metrics:
  forecastability: 0.7234
              cv: 1.2345
   missing_ratio: 0.0000

Unweighted Metrics:
  forecastability: 0.7234
              cv: 1.2345
   missing_ratio: 0.0000

  ✓ High forecastability (0.723)
  ✓ Excellent data quality (0.00% missing)
======================================================================
```

### Example 3: Compare Multiple Datasets

```python
from quito.utils.dataset_quality import compare_datasets, print_comparison

# Multiple datasets
datasets = {
    "Dataset_A": [np.random.randn(1000) for _ in range(30)],
    "Dataset_B": [np.random.randn(800) for _ in range(40)],
    "Dataset_C": [np.random.randn(1200) for _ in range(20)],
}

# Compare
summaries = compare_datasets(
    datasets,
    period=24,
    compute_adf=False
)

# Print comparison table
print_comparison(summaries, sort_by="QualityScore")
```

Output:
```
==========================================================================================
QUITO: Dataset Quality Comparison
==========================================================================================

Dataset              QualityScore Forecast   Season     Missing%   Length    
------------------------------------------------------------------------------------------
Dataset_C            0.7845       0.7234     0.6543     0.00       1200      
Dataset_A            0.7654       0.7123     0.6234     0.00       1000      
Dataset_B            0.7234       0.6890     0.5890     2.50       800       
==========================================================================================
QualityScore = 0.45*Forecast + 0.25*Season + 0.15*(1-Missing) + 0.15*LengthNorm
```

### Example 4: Plot Quality Distribution

```python
from quito.utils.dataset_quality import plot_forecastability_cdf

# Visualize forecastability distribution across datasets
plot_forecastability_cdf(
    datasets,
    period=24,
    save_path="quality_comparison.png"
)
```

### Example 5: Load from File

```python
from quito.utils.dataset_quality import evaluate_dataset_from_file

# Evaluate data from file (.npy, .csv, .txt)
results = evaluate_dataset_from_file(
    "data/timeseries.csv",
    period=24,
    verbose=True
)
```

## Running Example Scripts

Two example scripts are provided:

### Basic Analysis
```bash
python examples/data_analysis/analyze_dataset_quality.py
```

This demonstrates:
- Single series evaluation
- Dataset evaluation
- Loading and analyzing parquet data
- Interpreting quality metrics

### Dataset Comparison
```bash
python examples/data_analysis/compare_datasets_quality.py
```

This demonstrates:
- Comparing multiple datasets
- Quality score ranking
- Visualization
- Interpretation guidelines

## API Reference

### Core Functions

#### `evaluate_series()`
```python
evaluate_series(
    x: ArrayLike,                    # Time series data
    period: Optional[int] = None,     # Seasonal period
    fs: float = 1.0,                 # Sampling frequency
    compute_adf: bool = False,       # Compute ADF test
    adf_fill: str = 'mean',          # How to fill NaNs for ADF
    forecast_window: Optional[int] = None,  # Window for forecast
    fill_for_metrics: str = 'none'   # How to fill NaNs for metrics
) -> SeriesQuality
```

#### `evaluate_dataset()`
```python
evaluate_dataset(
    series_list: List[ArrayLike],
    period: Optional[int] = None,
    fs: float = 1.0,
    compute_adf: bool = False,
    forecast_window: Optional[int] = None,
    fill_for_metrics: str = 'none',
    verbose: bool = True
) -> Dict[str, Union[float, Dict]]
```

Returns weighted and unweighted metrics for the entire dataset.

#### `compare_datasets()`
```python
compare_datasets(
    datasets: Dict[str, List[ArrayLike]],
    period: Optional[int] = None,
    fs: float = 1.0,
    compute_adf: bool = False,
    forecast_window: Optional[int] = None
) -> Dict[str, Dict[str, float]]
```

Returns quality summaries with composite QualityScore for each dataset.

### Utility Functions

- `print_dataset_report(results)`: Pretty-print evaluation results
- `print_comparison(summaries)`: Pretty-print dataset comparison
- `plot_forecastability_cdf(datasets)`: Plot forecastability distribution
- `evaluate_dataset_from_file(path)`: Load and evaluate from file

## Best Practices

### 1. Data Selection
- Use QualityScore > 0.7 for primary training data
- Use 0.5-0.7 with augmentation or preprocessing
- Investigate < 0.5 for quality issues

### 2. Missing Data
- < 1%: Excellent
- 1-5%: Acceptable
- > 5%: Consider imputation or removal

### 3. Forecastability
- > 0.7: High confidence in forecasting
- 0.5-0.7: Moderate confidence
- < 0.5: Low confidence, consider alternative approaches

### 4. Seasonality
- Check if detected period matches expected (e.g., 24 for hourly with daily pattern)
- High seasonality (> 0.6) indicates strong patterns
- Low seasonality doesn't mean bad quality, just different patterns

### 5. Performance Tips
- Set `compute_adf=False` for faster evaluation
- Use `verbose=False` for programmatic use
- Process in batches for very large datasets

## References

This toolkit is inspired by:
- [Large-Time-Series-Model](https://github.com/thuml/Large-Time-Series-Model) repository
- [Timer/UTSD paper](https://arxiv.org/pdf/2402.02368): Unified Time Series Dataset evaluation

## Troubleshooting

### Seasonality returns NaN
- Install: `pip install statsmodels`
- Ensure `period` is valid: `2 ≤ period < series_length`

### ADF returns NaN
- Install: `pip install arch`
- Series may be too short or have too many NaNs

### Plotting fails
- Install: `pip install matplotlib`

## Next Steps

1. **Analyze your data**: Run `evaluate_dataset()` on your time series
2. **Compare datasets**: Use `compare_datasets()` to rank options
3. **Make decisions**: Use quality scores to guide data selection
4. **Preprocess**: Address quality issues before training
5. **Document**: Record quality metrics for reproducibility

For more examples, see:
- `examples/data_analysis/analyze_dataset_quality.py`
- `examples/data_analysis/compare_datasets_quality.py`

