# QUITO: Time Series Forecasting Framework

QUITO is a comprehensive time series forecasting framework that supports multiple state-of-the-art models including PatchTST, iTransformer, TSMixer, Crossformer, Pyraformer, and more. It provides a unified interface for training, fine-tuning, evaluation, and hyperparameter tuning of time series models.

## Features

- **Multiple Model Support**: PatchTST, iTransformer, TSMixer, Crossformer, Pyraformer, DLinear, TiRex, Chronos, TimesFM
- **Unified Interface**: Consistent API across all models through YAML configuration
- **Distributed Training**: Multi-GPU support via PyTorch DistributedDataParallel
- **Hyperparameter Tuning**: Built-in Ray Tune integration
- **Dataset Quality Analysis**: Comprehensive tools for evaluating time series data quality
- **Zero-Shot Inference**: Support for pre-trained foundation models (Chronos, TimesFM, TiRex)

## Installation

### Basic Installation

```bash
git clone https://github.com/alipay/quito-10b.git
cd quito-10b
pip install -r requirements.txt
```

### CLI Installation

Install QUITO with CLI support:

```bash
pip install -e .
```

This will install the `quito-cli` command for easy access to all training and evaluation scripts.

### Optional Dependencies

For zero-shot inference with foundation models:

```bash
# Chronos-2
pip install chronos-forecasting

# TimesFM-2.5
# Follow instructions at: https://github.com/google-research/timesfm/tree/master

# TiRex-Zero
# Follow instructions at: https://github.com/NX-AI/tirex/tree/main

# Dataset quality analysis
pip install statsmodels arch matplotlib
```

## Quick Start

### Using CLI

```bash
# Pre-training
quito-cli pretrain --config_path configs/pretrain/{model}/config.yaml

# Fine-tuning
quito-cli finetune --config_path configs/finetune/{model}/config.yaml

# Evaluation
quito-cli evaluate --config_path configs/evaluate/{model}/config.yaml --num_gpus 2

# Hyperparameter tuning
quito-cli tune --config_path configs/tune/{model}/config.yaml \
              --tuning_config_path configs/tune/{model}/tune_config.yaml \
              --num_workers 4 --num_samples 100
```

### Using Scripts Directly

```bash
# Pre-training with distributed training
torchrun --nproc_per_node 4 scripts/pretrain.py \
    --config_path configs/pretrain/patchtst/config.yaml

# Fine-tuning
torchrun --nproc_per_node 4 scripts/finetune.py \
    --config_path configs/finetune/patchtst/config.yaml

# Evaluation
python scripts/evaluate.py \
    --config_path configs/evaluate/patchtst/config.yaml \
    --num_gpus 2

# Hyperparameter tuning
python scripts/tune.py \
    --config_path configs/tune/patchtst/config.yaml \
    --tuning_config_path configs/tune/patchtst/tune_config.yaml \
    --num_workers 4 \
    --num_samples 100 \
    --use_gpu 1
```

## Configuration

All models use YAML configuration files. Example structure:

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

Configuration files are organized in `configs/`:
- `configs/pretrain/` - Pre-training configurations
- `configs/finetune/` - Fine-tuning configurations
- `configs/evaluate/` - Evaluation configurations
- `configs/tune/` - Hyperparameter tuning configurations

## Supported Models

### Trainable Models

- **PatchTST**: Patch-based transformer for long-term forecasting
- **iTransformer**: Inverted transformer architecture
- **TSMixer**: MLP-based time series model
- **Crossformer**: Cross-dimension attention for multivariate forecasting
- **Pyraformer**: Pyramidal attention mechanism
- **DLinear**: Simple linear model baseline
- **TSTrans former**: Classic transformer for time series

### Zero-Shot Inference Models

- **Chronos-2**: Amazon's pre-trained foundation model
- **TimesFM-2.5**: Google's time series foundation model
- **TiRex-Zero**: NX-AI's zero-shot forecasting model

Note: Zero-shot models are for inference only and cannot be fine-tuned.

## Workflow

### 1. Pre-training

Train a model from scratch on your pre-training dataset:

```bash
quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml
```

This trains the model on unlabeled time series data to learn general patterns.

### 2. Fine-tuning

Fine-tune a pre-trained model on specific downstream tasks:

```bash
quito-cli finetune --config_path configs/finetune/patchtst/config.yaml
```

Fine-tuning uses the TRAIN portion of your TRAIN/TEST split.

### 3. Hyperparameter Tuning

Optimize hyperparameters using TRAIN/VALID split:

```bash
quito-cli tune --config_path configs/tune/patchtst/config.yaml \
              --tuning_config_path configs/tune/patchtst/tune_config.yaml \
              --num_workers 4 \
              --num_samples 100
```

The tuning process uses Ray Tune for efficient hyperparameter search.

### 4. Evaluation

Evaluate model performance on test data:

```bash
quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml --num_gpus 2
```

Evaluation computes forecasting metrics (MSE, MAE, etc.) on the TEST set.

## Dataset Quality Analysis

QUITO includes comprehensive tools for analyzing time series dataset quality:

```bash
# Analyze individual dataset
python examples/analyze_dataset_quality.py

# Compare multiple datasets
python examples/compare_datasets_quality.py

# Analyze your own parquet files
python examples/analyze_open_hour_train_quality.py \
    --max_length 5000 \
    --max_series_per_file 50 \
    --sampling_strategy uniform
```

### Quality Metrics

- **Forecastability** (0-1): Predictability based on spectral entropy
- **Seasonality Strength** (0-1): Strength of seasonal patterns
- **Missing Data**: Percentage of missing values
- **Coefficient of Variation**: Relative variability
- **ADF Statistic**: Stationarity measure

See [docs/DATASET_QUALITY.md](docs/DATASET_QUALITY.md) for detailed information.

## Data Format

QUITO expects Parquet files with the following structure:

```
Required columns:
- timestamp: Time index
- value: Time series values

Optional columns:
- item_id: For multiple series in one file
```

Example dataset structure:

```
datasets/
└── parquet_data/
    └── open_hour_train/
        ├── hour_train_hour_p1.parquet
        ├── hour_train_hour_p2.parquet
        └── ...
```

Generate sample data:

```bash
python examples/create_data.py
```

## Multi-GPU Training

### Single Node, Multiple GPUs

```bash
# Using torchrun (recommended)
torchrun --nproc_per_node 4 scripts/pretrain.py \
    --config_path configs/pretrain/patchtst/config.yaml

# Or using quito-cli
CUDA_VISIBLE_DEVICES=0,1,2,3 quito-cli pretrain \
    --config_path configs/pretrain/patchtst/config.yaml
```

### Multi-Node Training

```bash
# Node 0 (master)
torchrun --nproc_per_node 4 --nnodes 2 --node_rank 0 \
    --master_addr master_ip --master_port 29500 \
    scripts/pretrain.py --config_path configs/pretrain/patchtst/config.yaml

# Node 1 (worker)
torchrun --nproc_per_node 4 --nnodes 2 --node_rank 1 \
    --master_addr master_ip --master_port 29500 \
    scripts/pretrain.py --config_path configs/pretrain/patchtst/config.yaml
```

## Examples

The `examples/` directory contains self-contained scripts:

- `create_data.py`: Generate synthetic time series data
- `analyze_dataset_quality.py`: Analyze dataset quality metrics
- `compare_datasets_quality.py`: Compare multiple datasets
- `analyze_open_hour_train_quality.py`: Analyze your own parquet files
- `cluster_items_by_quality.py`: Cluster time series by quality
- `build_cluster_files.py`: Build cluster-specific datasets

See [examples/README.md](examples/README.md) for detailed information.

## Documentation

- **[pretrain.md](docs/pretrain.md)**: Pre-training guide
- **[finetune.md](docs/finetune.md)**: Fine-tuning guide
- **[evaluate.md](docs/evaluate.md)**: Evaluation guide
- **[tune.md](docs/tune.md)**: Hyperparameter tuning guide
- **[DATASET_QUALITY.md](docs/DATASET_QUALITY.md)**: Dataset quality analysis guide

## Project Structure

```
quito-10b/
├── configs/              # YAML configuration files
│   ├── pretrain/        # Pre-training configs
│   ├── finetune/        # Fine-tuning configs
│   ├── evaluate/        # Evaluation configs
│   └── tune/            # Hyperparameter tuning configs
├── docs/                # Documentation
├── examples/            # Example scripts
├── quito/              # Core package
│   ├── config/         # Configuration classes
│   ├── datasets.py     # Dataset loading
│   ├── metrics.py      # Evaluation metrics
│   ├── models/         # Model implementations
│   ├── trainers/       # Training logic
│   └── utils/          # Utilities
├── scripts/            # Main training scripts
│   ├── pretrain.py    # Pre-training script
│   ├── finetune.py    # Fine-tuning script
│   ├── evaluate.py    # Evaluation script
│   └── tune.py        # Hyperparameter tuning script
├── cli.py             # Command-line interface
├── pyproject.toml     # Package configuration
└── README.md          # This file
```

## Common Use Cases

### 1. Train a Model from Scratch

```bash
# Create sample data
python examples/create_data.py

# Analyze data quality
python examples/analyze_dataset_quality.py

# Pre-train model
quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml
```

### 2. Fine-tune a Pre-trained Model

```bash
# Fine-tune on your specific task
quito-cli finetune --config_path configs/finetune/patchtst/config.yaml
```

### 3. Hyperparameter Optimization

```bash
# Tune hyperparameters
quito-cli tune --config_path configs/tune/patchtst/config.yaml \
              --tuning_config_path configs/tune/patchtst/tune_config.yaml \
              --num_workers 4 \
              --num_samples 100
```

### 4. Zero-Shot Inference

```bash
# Evaluate pre-trained foundation model
quito-cli evaluate --config_path configs/evaluate/chronos/config.yaml
```

### 5. Batch Evaluation

```bash
# Evaluate multiple configs
for config in configs/evaluate/*/config.yaml; do
    quito-cli evaluate --config_path $config --num_gpus 2
done
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError**
```bash
# Install missing dependencies
pip install -r requirements.txt
pip install -r requirements-optional.txt  # For foundation models
```

**CUDA Out of Memory**
- Reduce `batch_size` in config
- Reduce `seq_len` or `forecast_horizon`
- Use gradient accumulation
- Enable mixed precision training

**FileNotFoundError: parquet file not found**
```bash
# Generate sample data first
python examples/create_data.py
```

**"RuntimeError: element 0 of tensors does not require grad"**
- This occurs with zero-shot models (Chronos, TimesFM, TiRex)
- Use `evaluate` instead of `pretrain` or `finetune`

### Performance Tips

1. **Use appropriate batch size**: Start with 32 and adjust based on GPU memory
2. **Enable mixed precision**: Set `use_amp: true` in config
3. **Use multiple GPUs**: Leverage distributed training for faster training
4. **Analyze data quality first**: Use quality analysis tools before training
5. **Start with small models**: Test with DLinear or smaller models first

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## Citation

If you use QUITO in your research, please cite:

```bibtex
@software{quito2024,
  title={QUITO: Time Series Forecasting Framework},
  author={Alipay},
  year={2024},
  url={https://github.com/alipay/quito-10b}
}
```

## License

See LICENSE file for details.

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation in `docs/`
- Review examples in `examples/`

---

**Happy Forecasting! 📈**
