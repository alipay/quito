# QUITO Examples

This directory contains self-contained example scripts demonstrating various QUITO features.

## 🎯 Overview

All example scripts are **independent** and **self-contained** - no cross-dependencies between files.

## 📂 Example Scripts

### Training Examples (Trainable Models)

#### `train_patchtst.py`
Train the PatchTST model on your data.

```bash
python examples/train_patchtst.py
```

**Features**:
- Patch-based transformer architecture
- Works on CPU or GPU
- No external model dependencies
- Full training pipeline

**Config**: `configs/patchtst.yaml`

---

#### `train_huggingface.py`
Train any Hugging Face time series model.

```bash
python examples/train_huggingface.py
```

**Features**:
- Generic HuggingFace model wrapper
- Flexible architecture support
- GPU/CPU compatible
- Requires `transformers` (already in requirements.txt)

**Config**: `configs/huggingface.yaml`

---

### Evaluation Examples (Zero-Shot Inference)

#### `eval_chronos.py`
Evaluate Amazon Chronos model (zero-shot inference only).

```bash
python examples/eval_chronos.py
```

**Features**:
- Pre-trained foundation model
- Zero-shot forecasting
- No training required
- Requires: `pip install git+https://github.com/amazon-science/chronos-forecasting.git`

**Config**: `configs/chronos.yaml`

⚠️ **Note**: Chronos is inference-only and cannot be fine-tuned.

---

#### `eval_moriai.py`
Evaluate Salesforce Moirai model (zero-shot inference only).

```bash
python examples/eval_moriai.py
```

**Features**:
- Pre-trained foundation model
- Zero-shot forecasting
- No training required
- Requires: `pip install uni2ts`

**Config**: `configs/moriai.yaml`

⚠️ **Note**: Moirai is inference-only and cannot be fine-tuned.

---

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

## 📁 Configuration Files

All model configurations are in `configs/`:

```
configs/
├── patchtst.yaml      # PatchTST configuration
├── huggingface.yaml   # HuggingFace model config
├── chronos.yaml       # Chronos model config
└── moriai.yaml        # Moirai model config
```

### Configuration Structure

```yaml
data:
  common:
    seq_len: 512              # Input sequence length
    forecast_horizon: 96      # Prediction horizon
    features: "S"             # S: univariate, M: multivariate
    freq: "H"                 # H: hourly, D: daily, etc.

model:
  model_name: "patchtst"
  # Model-specific parameters

training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.001
```

## 🚀 Usage Patterns

### Quick Start
```bash
# 1. Generate sample data
python examples/create_data.py

# 2. Analyze data quality
python examples/analyze_dataset_quality.py

# 3. Train a model
python examples/train_patchtst.py

# 4. Evaluate zero-shot model
python examples/eval_chronos.py
```

### GPU Training
```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python examples/train_patchtst.py

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/train_patchtst.py
```

### CPU Training
All examples automatically fall back to CPU if no GPU is available:
```bash
python examples/train_patchtst.py  # Automatically uses CPU
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

### Add Custom Models
See [docs/EXAMPLES.md](../docs/tune.md) for detailed instructions on adding custom models.

## 📚 Documentation

For more detailed information:
- **Training Guide**: [docs/TRAINING.md](../docs/pretrain.md)
- **Evaluation Guide**: [docs/EVALUATION.md](../docs/evaluate.md)
- **Dataset Quality Guide**: [docs/DATASET_QUALITY.md](../docs/DATASET_QUALITY.md)
- **Examples Explained**: [docs/EXAMPLES.md](../docs/tune.md)

## 🐛 Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'chronos'"**
```bash
pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

**"ModuleNotFoundError: No module named 'uni2ts'"**
```bash
pip install uni2ts
```

**"ValueError: 'H' is not a valid Freq"**
- Make sure `freq: "H"` is uppercase in your config file

**"FileNotFoundError: parquet file not found"**
```bash
python examples/create_data.py  # Generate sample data first
```

**"RuntimeError: element 0 of tensors does not require grad"**
- Chronos and Moirai are inference-only models
- Use `eval_chronos.py` or `eval_moriai.py` instead of training scripts

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

## 🎓 Learning Path

1. **Beginner**: Start with `create_data.py` → `train_patchtst.py`
2. **Intermediate**: Try `analyze_dataset_quality.py` → customize configs
3. **Advanced**: Multi-GPU training → zero-shot evaluation → model comparison

---

All examples are production-ready and can be adapted for your specific use case! 🚀

