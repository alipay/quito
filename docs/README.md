# QUITO Documentation

Welcome to the QUITO documentation! This directory contains comprehensive guides for using QUITO.

## 📚 Available Guides

### [TRAINING.md](TRAINING.md) - Model Training Guide
Complete guide for training time series foundation models:
- Supported trainable models (PatchTST, DLinear, HuggingFace)
- Single GPU, multi-GPU (DDP), and CPU training
- Configuration management
- Data preparation
- Hyperparameter tuning
- Troubleshooting common issues

**Start here if you want to train models on your data.**

### [EVALUATION.md](EVALUATION.md) - Model Evaluation Guide
Guide for evaluating and comparing models:
- Zero-shot inference (Chronos, Moirai)
- Evaluation metrics (MSE, MAE, etc.)
- Model comparison strategies
- Custom evaluation scripts
- Benchmark results

**Start here if you want to evaluate pre-trained models.**

### [EXAMPLES.md](EXAMPLES.md) - Example Scripts Explained
Detailed explanations of all example scripts:
- Training examples (`train_patchtst.py`, `train_huggingface.py`)
- Evaluation examples (`eval_chronos.py`, `eval_moriai.py`)
- Data generation (`create_data.py`)
- Dataset quality analysis examples
- Configuration files walkthrough

**Start here if you want to understand how the examples work.**

### [DATASET_QUALITY.md](DATASET_QUALITY.md) - Dataset Quality Analysis
Comprehensive guide for evaluating time series dataset quality:
- Quality metrics (forecastability, seasonality, etc.)
- Per-series and dataset-level analysis
- Cross-dataset comparison with QualityScore
- Visualization tools
- API reference and usage examples
- Best practices for data selection

**Start here if you want to analyze and compare datasets.**

## 🎯 Quick Start by Goal

| I want to... | Read this guide... |
|--------------|-------------------|
| Train a model on my data | [TRAINING.md](TRAINING.md) |
| Evaluate a pre-trained model | [EVALUATION.md](EVALUATION.md) |
| Understand the example scripts | [EXAMPLES.md](EXAMPLES.md) |
| Check if my dataset is good for forecasting | [DATASET_QUALITY.md](DATASET_QUALITY.md) |
| Compare multiple datasets | [DATASET_QUALITY.md](DATASET_QUALITY.md) |
| Use multiple GPUs | [TRAINING.md](TRAINING.md) → Multi-GPU Training |
| Troubleshoot an error | [TRAINING.md](TRAINING.md) → Troubleshooting |
| Add a custom model | [EXAMPLES.md](EXAMPLES.md) → Adding Custom Models |

## 📖 Documentation Structure

```
docs/
├── README.md              # This file - documentation index
├── TRAINING.md            # Training guide
├── EVALUATION.md          # Evaluation guide
├── EXAMPLES.md            # Example scripts explained
└── DATASET_QUALITY.md     # Dataset quality analysis
```

## 💡 Tips for Using This Documentation

1. **Start with the goal-based quick start** above
2. **Follow code examples** - all guides include runnable code
3. **Check troubleshooting sections** if you encounter errors
4. **Refer back to examples** in `examples/` directory for full implementations

## 🔗 Additional Resources

- **Main README**: [`../README.md`](../README.md) - Project overview and quick start
- **Example Scripts**: [`../examples/`](../examples/) - Self-contained runnable examples
- **Configuration Files**: [`../examples/configs/`](../examples/configs/) - Model configs
- **Requirements**: [`../requirements-optional.txt`](../requirements-optional.txt) - Optional dependencies

## 📝 Contributing to Documentation

If you find errors or have suggestions for improving the documentation:
1. Check existing documentation for similar content
2. Follow the existing format and style
3. Include code examples when possible
4. Test all code examples before submitting

## 🆘 Getting Help

If you're stuck after reading the documentation:
1. Check the [Troubleshooting](TRAINING.md#troubleshooting) section
2. Review [Example Scripts](../examples/) for working implementations
3. Open an issue with:
   - What you're trying to do
   - What error you're seeing
   - What documentation you've already read

---

Happy forecasting with QUITO! 🚀

