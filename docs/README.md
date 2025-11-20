# QUITO Documentation

Welcome to the QUITO documentation! This directory contains comprehensive guides for using the QUITO library.

## 📚 Documentation Index

### Getting Started

- **[Training Guide](TRAINING_GUIDE.md)** - Complete guide to training time series models
  - Installation and setup
  - Data preparation
  - Model training
  - GPU optimization
  - Troubleshooting

- **[Configuration Quick Start](CONFIG_QUICK_START.md)** - 5-minute guide to configurations
  - Using pre-made configs
  - Overriding values
  - Creating custom configs
  - Common patterns

### Detailed Guides

- **[Configuration Guide](CONFIGS_GUIDE.md)** - Comprehensive configuration documentation
  - All available config options
  - Configuration structure
  - Multiple configuration examples
  - Best practices

- **[Scripts Guide](SCRIPTS_GUIDE.md)** - Training scripts documentation
  - `train.py` - Main training script
  - `extract_sample_data.py` - Data sampling tool
  - `demo_config.py` - Configuration examples
  - `test_training.py` - Testing utilities

## 🚀 Quick Navigation

### I want to...

**Train a model quickly**
→ Start with [Training Guide](TRAINING_GUIDE.md) → Quick Start section

**Understand configurations**
→ Read [Configuration Quick Start](CONFIG_QUICK_START.md)

**Use GPU training**
→ See [Training Guide](TRAINING_GUIDE.md) → GPU Training section

**Create custom configs**
→ Check [Configuration Guide](CONFIGS_GUIDE.md) → Creating Custom Configurations

**Understand training scripts**
→ Read [Scripts Guide](SCRIPTS_GUIDE.md)

**Troubleshoot issues**
→ See [Training Guide](TRAINING_GUIDE.md) → Troubleshooting section

## 📖 Documentation Overview

### Training Guide (TRAINING_GUIDE.md)
The main guide covering:
- Installation
- Data preparation and sampling
- Training script usage
- GPU training and optimization
- Configuration options
- Examples and troubleshooting

### Configuration Quick Start (CONFIG_QUICK_START.md)
A quick 5-minute tutorial covering:
- Using pre-made configurations
- Overriding config values
- Common configuration patterns
- Command-line examples

### Configuration Guide (CONFIGS_GUIDE.md)
Comprehensive configuration documentation:
- Available configuration files
- Configuration structure
- All options explained
- Creating custom configs
- Configuration tips and tricks

### Scripts Guide (SCRIPTS_GUIDE.md)
Detailed scripts documentation:
- Training script (`train.py`)
- Data extraction (`extract_sample_data.py`)
- Configuration demo (`demo_config.py`)
- Testing script (`test_training.py`)

## 🎯 Common Tasks

### Training Your First Model

1. **Extract sample data**:
```bash
python scripts/extract_sample_data.py \
    --input-dir data/open_hour_train \
    --output-dir data/sample_train \
    --sample-size 10000
```

2. **Train with default config**:
```bash
python scripts/train.py --config configs/train_config.yaml
```

See [Training Guide](TRAINING_GUIDE.md) for more details.

### GPU Training

```bash
# Use GPU-optimized config
python scripts/train.py --config configs/pyraformer_gpu.yaml

# Or enable FP16
python scripts/train.py --config configs/train_config.yaml --fp16
```

See [Training Guide](TRAINING_GUIDE.md) → GPU Training for details.

### Creating Custom Configurations

```bash
# Copy template
cp configs/train_config.yaml configs/my_config.yaml

# Edit your config
# Then train
python scripts/train.py --config configs/my_config.yaml
```

See [Configuration Guide](CONFIGS_GUIDE.md) for all options.

## 📁 Related Directories

- **`configs/`** - YAML configuration files
  - `train_config.yaml` - Default template
  - `pyraformer_gpu.yaml` - GPU training
  - `informer_quick.yaml` - Quick testing
  - `patchtst_production.yaml` - Production settings

- **`scripts/`** - Training and utility scripts
  - `train.py` - Main training script
  - `extract_sample_data.py` - Data sampling
  - `demo_config.py` - Config examples
  - `test_training.py` - Testing

- **`examples/`** - Example usage scripts
  - `run_pyraformer.py` - Pyraformer example
  - `setup_data_folder.py` - Data setup example

## 🆘 Getting Help

### Documentation Not Clear?
- Check the [Training Guide](TRAINING_GUIDE.md) for comprehensive explanations
- Read the [Configuration Quick Start](CONFIG_QUICK_START.md) for a quick tutorial
- Review examples in the [Scripts Guide](SCRIPTS_GUIDE.md)

### Found a Bug?
- Open an issue on GitHub with:
  - Error message
  - Your configuration file
  - Steps to reproduce

### Want to Contribute?
- Improve documentation
- Add examples
- Report issues
- Submit pull requests

## 📝 Documentation Guidelines

When adding new documentation:
1. Place all `.md` files in this `docs/` folder
2. Update this README with links to new docs
3. Use clear headings and examples
4. Include code snippets where helpful
5. Link between related documents

## 🔄 Keep Documentation Updated

When making changes to the codebase:
- Update relevant documentation
- Add new examples if needed
- Update configuration guides for new options
- Keep the main README.md in sync

---

**Quick Links:**
- [Main README](../README.md)
- [Training Guide](TRAINING_GUIDE.md)
- [Configuration Quick Start](CONFIG_QUICK_START.md)
- [Configuration Guide](CONFIGS_GUIDE.md)
- [Scripts Guide](SCRIPTS_GUIDE.md)

