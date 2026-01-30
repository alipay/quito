# QUITO Examples

This directory contains self-contained example scripts demonstrating various QUITO features.

## 🎯 Overview

All example scripts are **independent** and **self-contained** - no cross-dependencies between files.

## 📂 Example Scripts

### Data Utilities

#### `create_data.py`
Generate synthetic time series data for testing.

```bash
python examples/create_data.py
```

**Outputs**:
- Creates `datasets/parquet_data/open_hour_train/hour_train_hour_p1.parquet`
- Synthetic hourly data with trend and seasonality
- Ready for training/evaluation

---

### Dataset Quality Analysis

#### `analyze_dataset_quality.py`
Analyze time series dataset quality with comprehensive metrics.

```bash
python examples/analyze_dataset_quality.py
```

**Features**:
- Forecastability analysis (spectral entropy)
- Seasonality strength (STL decomposition)
- Missing data analysis
- Statistical properties (CV, ADF)
- Per-series and dataset-level metrics

**Requirements** (optional):
```bash
pip install statsmodels arch matplotlib
```

**Outputs**:
- Quality metrics for individual series
- Dataset-level summary statistics
- Interpretation and recommendations

---

#### `compare_datasets_quality.py`
Compare quality across multiple time series datasets.

```bash
python examples/compare_datasets_quality.py
```

**Features**:
- Cross-dataset quality comparison
- Composite QualityScore ranking
- CDF visualization
- Synthetic datasets with different characteristics

**Requirements** (optional):
```bash
pip install statsmodels matplotlib
```

**Outputs**:
- Quality comparison table
- Ranked datasets by QualityScore
- Forecastability CDF plot (`dataset_quality_comparison.png`)

---

#### `analyze_open_hour_train_quality.py`
Analyze dataset quality for your own parquet files in `open_hour_train/` directory.

```bash
# Analyze all files with default settings
python examples/analyze_open_hour_train_quality.py

# Custom truncation and sampling
python examples/analyze_open_hour_train_quality.py \
    --max_length 5000 \
    --max_series_per_file 50 \
    --sampling_strategy uniform

# Analyze specific files
python examples/analyze_open_hour_train_quality.py \
    --files hour_train_hour_p1.parquet hour_train_hour_p2.parquet
```

**Features**:
- Handles large-scale series with truncation/sampling
- Automatic column detection (`value`, `ind_1`, or first numeric column)
- Supports multiple series per file (via `item_id`)
- Cross-file comparison
- Configurable sampling strategies

**Key Options**:
- `--max_length`: Truncate long series (default: 10000)
- `--max_series_per_file`: Sample series per file (default: 100)
- `--sampling_strategy`: `random`, `first`, `last`, or `uniform` (default: random)
- `--period`: Seasonal period for hourly data (default: 24)
- `--compute_adf`: Include ADF stationarity test

**Requirements** (optional):
```bash
pip install statsmodels arch  # For full features
```

**Outputs**:
- Per-file quality reports
- Cross-file comparison table (if multiple files)
- Quality metrics for each dataset

---

### Additional Utilities

#### `aggregate_results.py`
Aggregate and analyze results from multiple experiments.

```bash
python examples/aggregate_results.py
```

#### `cluster_items_by_quality.py`
Cluster time series items by their quality metrics.

```bash
python examples/cluster_items_by_quality.py
```

#### `build_cluster_files.py`
Build cluster-specific dataset files.

```bash
python examples/build_cluster_files.py
```

#### `merge_train_valid_test.py`
Merge train, validation, and test datasets.

```bash
python examples/merge_train_valid_test.py
```

#### `run_quality_analysis.sh`
Shell script to run quality analysis in batch mode.

```bash
bash examples/run_quality_analysis.sh
```

#### `run_quality_analysis_fast.sh`
Shell script to run fast quality analysis.

```bash
bash examples/run_quality_analysis_fast.sh
```

---

## 📁 Configuration Files

All model configurations are in `configs/`:

```
configs/
├── pretrain/          # Pre-training configurations
├── finetune/          # Fine-tuning configurations  
├── evaluate/          # Evaluation configurations
└── tune/              # Hyperparameter tuning configurations
```

### Configuration Structure

```yaml
data:
  common:
    seq_len: 512              # Input sequence length
    forecast_horizon: 96      # Prediction horizon
    features: "S"             # S: univariate, M: multivariate
    freq: "H"                 # H: hourly, D: daily, etc.

datasets:
  - dataset_name: "my_dataset"
    file_name: "datasets/parquet_data/open_hour_train/data.parquet"

model:
  model_name: "patchtst"
  # Model-specific parameters

training:
  task_type: "pretrain"       # pretrain, finetune, evaluate
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.001
  device: "cuda"
  num_gpus: 1
```

## 🚀 Usage Patterns

### Quick Start
```bash
# 1. Generate sample data
python examples/create_data.py

# 2. Analyze data quality
python examples/analyze_dataset_quality.py

# 3. Pre-train a model
quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml

# 4. Fine-tune a model
quito-cli finetune --config_path configs/finetune/patchtst/config.yaml

# 5. Evaluate a model
quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml
```

## 📊 Dataset Structure

Expected data format: **Parquet files** with columns:
- `timestamp`: Time index
- `value`: Time series values
- `item_id`: (Optional) For multiple series

Example location:
```
datasets/
└── parquet_data/
    └── open_hour_train/
        └── hour_train_hour_p1.parquet
```

## 🔧 Customization

### Modify Configuration
Edit YAML files in `configs/`:
```yaml
training:
  num_epochs: 20        # Increase epochs
  batch_size: 64        # Larger batch size
  learning_rate: 0.0001 # Lower learning rate
```

### Use Your Own Data
Update the data path in configs:
```yaml
datasets:
  - dataset_name: "my_dataset"
    file_name: "datasets/parquet_data/my_data/my_file.parquet"
```

## 📚 Documentation

For more detailed information:
- **Pre-training Guide**: [docs/pretrain.md](../docs/pretrain.md)
- **Fine-tuning Guide**: [docs/finetune.md](../docs/finetune.md)
- **Evaluation Guide**: [docs/evaluate.md](../docs/evaluate.md)
- **Hyperparameter Tuning Guide**: [docs/tune.md](../docs/tune.md)
- **Dataset Quality Guide**: [docs/DATASET_QUALITY.md](../docs/DATASET_QUALITY.md)

## 🐛 Troubleshooting

### Dataset Quality Analysis Issues

**"Seasonality strength returns NaN"**
```bash
pip install statsmodels  # Required for STL decomposition
```

**"ADF statistic returns NaN"**
```bash
pip install arch  # Required for ADF test
```

**"Plotting fails"**
```bash
pip install matplotlib  # Required for visualization
```

## 💡 Best Practices

 1. **Start with data generation**: Run `create_data.py` first
 2. **Check data quality**: Use `analyze_dataset_quality.py` before training
 3. **Start small**: Use small configs for testing, then scale up
 4. **Monitor training**: Check logs for convergence
 5. **Compare models**: Evaluate multiple models on the same data

---

All examples are production-ready and can be adapted for your specific use case! 🚀

