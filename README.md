# QUITO-10B: A High Quality CloudOps Time Series Dataset with 10 Billion Tokens

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](docs/)

**QUITO** (**Q**uality Cl**U**d**O**ps **T**ime Series) is a comprehensive library for large-scale time series model training and evaluation on CloudOps data with 10 billion tokens. Designed with the same philosophy as Hugging Face Transformers, QUITO provides a unified interface for training, fine-tuning, and evaluating state-of-the-art time series models across various operational domains.

## 🚀 Features

- **10 Billion Token Dataset**: High-quality CloudOps time series data at scale
- **Unified Model Interface**: Consistent API for all time series models
- **Large-Scale Training**: GPU support with mixed precision (FP16) training
- **Configuration-Driven**: OmegaConf-based YAML configuration system
- **Comprehensive Evaluation**: Built-in metrics and benchmarks inspired by TSGBench
- **CloudOps Focused**: Optimized for operational metrics, monitoring, and forecasting
- **Production Ready**: Battle-tested on real-world CloudOps data

## 📦 Installation

```bash
pip install quito
```

For development installation:
```bash
git clone https://github.com/iLampard/QUITO.git
cd QUITO
pip install -e .
```

## 🎯 Quick Start

### Training with Configuration Files

```bash
# Extract sample data
python scripts/extract_sample_data.py \
    --input-dir data/open_hour_train \
    --output-dir data/sample_train \
    --sample-size 10000

# Train with config file
python scripts/train.py --config configs/train_config.yaml

# Train with GPU and mixed precision
python scripts/train.py --config configs/pyraformer_gpu.yaml --fp16
```

### Python API Usage

```python
from quito import ModelConfig, TrainingConfig, load_dataset, ForecastingTrainer
from quito.models import AutoModelForTimeSeriesForecasting
from quito.config import ModelType

# Load dataset
train_dataset = load_dataset(
    data_path="data/sample_train",
    seq_len=96,
    pred_len=96,
    features="MS",
    scale=True
)

# Create model
model_config = ModelConfig(
    model_type=ModelType.PYRAFORMER,
    seq_len=96,
    pred_len=96,
    enc_in=5,
    d_model=512,
    n_heads=8
)
model = AutoModelForTimeSeriesForecasting.from_config(model_config)

# Train
training_config = TrainingConfig(num_epochs=10, per_device_train_batch_size=32)
trainer = ForecastingTrainer(model, train_dataset, config=training_config)
trainer.train()
```

## 📚 Documentation

- **[Training Guide](docs/TRAINING_GUIDE.md)** - Complete guide to training models
- **[Configuration Guide](docs/CONFIGS_GUIDE.md)** - YAML configuration documentation  
- **[Quick Start](docs/CONFIG_QUICK_START.md)** - 5-minute configuration tutorial
- **[Scripts Guide](docs/SCRIPTS_GUIDE.md)** - Training scripts documentation

## 🏗️ Project Structure

```
QUITO/
├── quito/                  # Main package
│   ├── models/            # Model implementations (Pyraformer, Informer, etc.)
│   ├── trainers/          # Training loops with GPU support
│   ├── config/            # Configuration classes
│   ├── datasets.py        # Data loading and preprocessing
│   └── utils/             # Utility functions
├── configs/               # YAML training configurations
│   ├── train_config.yaml
│   ├── pyraformer_gpu.yaml
│   └── ...
├── scripts/               # Training and utility scripts
│   ├── train.py          # Main training script
│   ├── extract_sample_data.py
│   └── ...
├── docs/                  # Documentation
│   ├── TRAINING_GUIDE.md
│   ├── CONFIGS_GUIDE.md
│   └── ...
└── examples/              # Example scripts
```

## 🎯 Supported Models

### Currently Implemented
- **Transformer-based**: Pyraformer, Informer, Autoformer, PatchTST

### Coming Soon
- **GAN-based**: TimeGAN, DoppelGANger, RGAN
- **VAE-based**: VQVAE, KVAE
- **Flow-based**: CTFP, Fourier Flow
- **ODE-based**: Neural ODE, ODE-RNN, Neural SDE

## 📊 Evaluation Metrics

QUITO includes comprehensive evaluation metrics inspired by TSGBench:

- **Model-based**: Discriminative Score, Predictive Score, Contextual-FID
- **Feature-based**: Marginal Distribution Difference, AutoCorrelation Difference
- **Distance-based**: Euclidean Distance, Dynamic Time Warping
- **Visualization**: t-SNE, Distribution Plots

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🌟 About QUITO-10B

**QUITO-10B** represents a high-quality CloudOps time series dataset with 10 billion tokens, providing a comprehensive resource for training and evaluating time series models on operational data at scale. The library is designed to handle large-scale CloudOps metrics, logs, and monitoring data for forecasting, anomaly detection, and operational intelligence.

### Dataset Scale
- **10 Billion Tokens**: Massive scale for robust model training
- **CloudOps Domain**: Real-world operational metrics and monitoring data
- **High Quality**: Carefully curated and preprocessed time series data
- **Production Scale**: Enterprise-grade data volume and complexity

## 🙏 Acknowledgments

- Inspired by [Hugging Face Transformers](https://github.com/huggingface/transformers)
- Evaluation metrics based on [TSGBench](https://github.com/YihaoAng/TSGBench)
- Model implementations inspired by [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- Built for real-world CloudOps and operational intelligence use cases

## 📞 Contact

- GitHub Issues: [Report bugs or request features](https://github.com/your-username/QUITO/issues)
- Discussions: [Join the community](https://github.com/your-username/QUITO/discussions)