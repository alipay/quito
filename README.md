# QUITO - Time Series Foundation Model Training

A lightweight framework for training and evaluating time series foundation models.

## Quick Start

### 1. Setup Environment
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install specific model dependencies (only if you need them)
# For Chronos:
pip install git+https://github.com/amazon-science/chronos-forecasting.git

# For Moirai:
pip install uni2ts

# See requirements-optional.txt for all optional dependencies
```

### 2. Create Sample Data
```bash
python examples/create_data.py
```

### 3. Train a Model

**Option A: Use Example Scripts (Recommended)**
```bash
# Training examples (self-contained, no arguments needed)
python examples/train_patchtst.py
python examples/train_huggingface.py

# Inference-only (zero-shot evaluation)
python examples/eval_chronos.py  # Chronos - zero-shot inference
python examples/eval_moriai.py   # Moirai - zero-shot inference
```

**Option B: Use General Script**
```bash
# Requires config path argument
python scripts/train.py --config_path examples/configs/chronos.yaml
```

## Project Structure

```
quito-10b/
├── examples/                       # Self-contained example scripts
│   ├── train_patchtst.py          # Train PatchTST (trainable)
│   ├── train_huggingface.py       # Train HF models (trainable)
│   ├── eval_chronos.py            # Evaluate Chronos (zero-shot)
│   ├── eval_moriai.py             # Evaluate Moirai (zero-shot)
│   ├── create_data.py             # Generate test data
│   ├── analyze_dataset_quality.py # Analyze dataset quality
│   ├── compare_datasets_quality.py # Compare multiple datasets
│   ├── configs/                   # Model configurations
│   │   ├── patchtst.yaml
│   │   ├── huggingface.yaml
│   │   ├── chronos.yaml
│   │   └── moriai.yaml
│   └── datasets/                  # Data storage
│       └── parquet_data/          # Parquet format data
├── scripts/                       # General training scripts
│   ├── train.py                   # Main training script (any config)
│   ├── evaluate.py                # Evaluation script
│   └── pretrain.py                # Legacy script
├── quito/                         # Core library
│   ├── models/                    # Model implementations
│   ├── config/                    # Configuration classes
│   ├── trainers/                  # Training logic
│   ├── utils/                     # Utilities
│   │   └── dataset_quality.py    # Dataset quality analysis
│   └── datasets.py                # Data loading
├── docs/                          # Documentation
│   ├── TRAINING.md                # Training guide
│   ├── EVALUATION.md              # Evaluation guide
│   ├── EXAMPLES.md                # Examples explained
│   └── DATASET_QUALITY.md         # Dataset quality analysis
├── configs/                       # Default configurations
│   └── config.yaml
└── requirements-optional.txt      # Optional model dependencies
```

## Supported Models

- **Chronos** - Amazon's time series foundation model (requires: `chronos`) ✅ **Trainable & Inference**
- **Moirai** - Salesforce's time series foundation model (requires: `uni2ts`) ⚠️ **Inference only**
- **HuggingFace** - Any HF time series model (requires: `transformers` - included)
- **PatchTST** - Patch-based transformer (no extra dependencies) ✅ **Trainable**
- **DLinear** - Decomposition linear model (no extra dependencies) ✅ **Trainable**

> **Note**: Foundation models like Chronos and Moirai require additional packages.  
> See `requirements-optional.txt` for installation instructions.
> 
> ⚠️ **Training Support**: Moirai is pre-trained zero-shot model for inference only.
> For training, use Chronos (fine-tuning), PatchTST, DLinear, or HuggingFace models.

## Key Features

✅ **Simple**: Each example is self-contained (~100 lines)  
✅ **Independent**: No cross-dependencies between examples  
✅ **GPU Ready**: Automatic GPU/CPU detection  
✅ **Lightweight**: Minimal dependencies  
✅ **Flexible**: Easy to customize configs  
✅ **Production**: Supports multi-GPU training  
✨ **Dataset Quality Analysis**: Comprehensive toolkit for evaluating time series quality  

## GPU Training

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python examples/train_chronos.py

# Multi-GPU with DDP
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/train_chronos.py
```

## Configuration

Edit YAML files in `examples/configs/` or `configs/`:

```yaml
data:
  common:
    seq_len: 512              # Input length
    forecast_horizon: 96      # Prediction length
    features: "S"             # S: univariate, M: multivariate

model:
  model_name: "chronos"
  pretrained_model_name_or_path: "amazon/chronos-t5-small"

training:
  num_epochs: 5
  batch_size: 8
  learning_rate: 0.0001
```

## Dataset Quality Analysis

QUITO includes a comprehensive toolkit for evaluating time series dataset quality before training:

```bash
# Analyze a single dataset
python examples/analyze_dataset_quality.py

# Compare multiple datasets
python examples/compare_datasets_quality.py
```

**Key Metrics**:
- 📊 **Forecastability** (0-1): How predictable is the series?
- 🔄 **Seasonality Strength** (0-1): Strength of seasonal patterns
- ❌ **Missing Ratio**: Fraction of missing data
- 📏 **Effective Length**: Non-NaN data points
- 🎯 **Quality Score**: Composite metric for ranking datasets

**Programmatic Usage**:
```python
from quito.utils.dataset_quality import evaluate_dataset, compare_datasets

# Evaluate single dataset
results = evaluate_dataset(series_list, period=24)
print(f"Forecastability: {results['weighted_metrics']['forecastability']:.3f}")

# Compare multiple datasets
summaries = compare_datasets({
    "Dataset_A": series_list_a,
    "Dataset_B": series_list_b,
})
```

**Quality Score Interpretation**:
- **0.8-1.0**: Excellent - Use for primary training
- **0.6-0.8**: Good - Suitable for most tasks
- **0.4-0.6**: Fair - Consider preprocessing
- **0.0-0.4**: Poor - Investigate quality issues

See **[docs/DATASET_QUALITY.md](docs/DATASET_QUALITY.md)** for full documentation.

## Dependencies

**Core Dependencies** (in `requirements.txt`):
- PyTorch, NumPy, Pandas, Transformers (required for all models)

**Optional Model Dependencies** (in `requirements-optional.txt`):
- **Chronos**: `pip install git+https://github.com/amazon-science/chronos-forecasting.git`
- **Moirai**: `pip install uni2ts`

**Optional Analysis Dependencies** (in `requirements-optional.txt`):
- **Dataset Quality (full features)**: `pip install statsmodels arch matplotlib`
  - Core metrics work with base dependencies
  - Optional: `statsmodels` (seasonality), `arch` (ADF test), `matplotlib` (plotting)

> 💡 **Tip**: Only install the optional dependencies for features you plan to use!

## Documentation

### 📚 Comprehensive Guides in `docs/`

- **[docs/TRAINING.md](docs/TRAINING.md)** - Complete training guide
  - Trainable models overview
  - Training scripts usage (single/multi-GPU/CPU)
  - Configuration options
  - Troubleshooting

- **[docs/EVALUATION.md](docs/EVALUATION.md)** - Evaluation guide
  - Zero-shot inference (Chronos, Moirai)
  - Evaluation metrics
  - Model comparison
  - Custom evaluation

- **[docs/EXAMPLES.md](docs/EXAMPLES.md)** - Example scripts explained
  - Training examples (`train_*.py`)
  - Evaluation examples (`eval_*.py`)
  - Configuration files
  - Common patterns

- **[docs/DATASET_QUALITY.md](docs/DATASET_QUALITY.md)** - Dataset quality analysis
  - Quality metrics explained
  - Forecastability and seasonality
  - Cross-dataset comparison
  - API reference and examples

### 🎯 Quick Navigation

| I want to... | Go to... |
|--------------|----------|
| Train a model on my data | [docs/TRAINING.md](docs/TRAINING.md) |
| Use pre-trained models (zero-shot) | [docs/EVALUATION.md](docs/EVALUATION.md) |
| Understand example scripts | [docs/EXAMPLES.md](docs/EXAMPLES.md) |
| Evaluate dataset quality | [docs/DATASET_QUALITY.md](docs/DATASET_QUALITY.md) |
| Use multiple GPUs | [docs/TRAINING.md](docs/TRAINING.md) → Multi-GPU Training |
| Troubleshoot errors | [docs/TRAINING.md](docs/TRAINING.md) → Troubleshooting |

## Design Philosophy

**Examples folder**: Self-contained scripts for specific models. Each script is independent and can run without any other example files.

**Scripts folder**: General-purpose scripts that work with any configuration file.

**Quito library**: Core functionality shared by all scripts.

Simple, clean, and production-ready! 🚀
