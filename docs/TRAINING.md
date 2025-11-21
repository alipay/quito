# Training Guide

Complete guide to training time series models with QUITO.

## Table of Contents

- [Overview](#overview)
- [Trainable Models](#trainable-models)
- [Training Scripts](#training-scripts)
- [Configuration](#configuration)
- [Single GPU Training](#single-gpu-training)
- [Multi-GPU Training](#multi-gpu-training)
- [CPU Training](#cpu-training)
- [Monitoring Training](#monitoring-training)
- [Troubleshooting](#troubleshooting)

## Overview

QUITO supports training time series models on your custom datasets. The framework provides:
- Simple, self-contained training scripts
- Automatic GPU/CPU detection
- Distributed training (DDP) support
- Easy configuration via YAML files
- Checkpoint saving and resuming

## Trainable Models

| Model | Type | Parameters | Best For |
|-------|------|------------|----------|
| **PatchTST** | Transformer | ~1M-10M | General purpose, long sequences |
| **DLinear** | Linear | ~10K-100K | Fast training, simple patterns |
| **HuggingFace** | Various | Varies | Specific HF models |

**Note**: Chronos and Moirai are inference-only. See [EVALUATION.md](EVALUATION.md) for details.

## Training Scripts

### Example Scripts (Recommended)

Located in `examples/`, these are self-contained and require no command-line arguments:

```bash
# PatchTST - Transformer-based model
python examples/train_patchtst.py

# HuggingFace models
python examples/train_huggingface.py
```

**Advantages:**
- ✅ No command-line arguments needed
- ✅ Self-contained with all imports
- ✅ Easy to customize
- ✅ Perfect for learning

### General Script

Located in `scripts/train.py`, works with any config file:

```bash
# Train with any config
python scripts/train.py --config_path examples/configs/patchtst.yaml

# Or use main config
python scripts/train.py --config_path configs/config.yaml
```

**Advantages:**
- ✅ Flexible - works with any config
- ✅ Production-ready
- ✅ Supports all training modes

## Configuration

### Config File Structure

Example: `examples/configs/patchtst.yaml`

```yaml
data:
  common:
    data_dir: "examples/datasets/parquet_data"
    seq_len: 512                    # Input sequence length
    forecast_horizon: 96            # Prediction length
    decoder_label_len: 48           # Decoder input length
    features: "S"                   # S: univariate, M: multivariate
    normalize: True                 # Normalize data
  
  datasets:
    EXAMPLE_DATA:
      file_name: "open_hour_train/hour_train_hour_p1.parquet"
      train_ratio: 0.7
      valid_ratio: 0.2
      test_ratio: 0.1
      freq: 'H'                     # H: hourly, D: daily, etc.
      target: 'value'

model:
  model_name: "patchtst"
  d_model: 128
  n_heads: 8
  e_layers: 3
  d_ff: 256
  dropout: 0.1
  patch_len: 16
  stride: 8

training:
  num_epochs: 10
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.0001
  warmup_ratio: 0.1
  gradient_accumulation_steps: 1
  max_grad_norm: 1.0
  seed: 42
  
  # Optimization
  fp16: False                       # Use mixed precision
  num_workers: 2
  scheduler: "cosine"
  optimizer: "adamw"

checkpointing:
  save_last_k: 3                    # Keep last 3 checkpoints
  save_epochs: 1                    # Save every N epochs

logging:
  output_dir: "outputs/patchtst_example"
  logging_steps: 50
  logging_epochs: 1

early_stopping:
  enable_early_stopping: True
  early_stopping_patience: 5
  es_metric: "mse"
  greater_is_better: False

evaluation:
  eval_metrics: ['mse', 'mae']
  eval_epochs: 1
```

### Key Configuration Options

**Data Configuration:**
- `seq_len`: Length of input sequence (e.g., 512 hours)
- `forecast_horizon`: How many steps to predict (e.g., 96 hours)
- `features`: "S" for univariate, "M" for multivariate
- `normalize`: Whether to normalize data (recommended: True)

**Model Configuration:**
- `d_model`: Model hidden dimension
- `n_heads`: Number of attention heads
- `e_layers`: Number of encoder layers
- `dropout`: Dropout rate for regularization

**Training Configuration:**
- `num_epochs`: Number of training epochs
- `batch_size`: Batch size (adjust based on GPU memory)
- `learning_rate`: Learning rate (typical: 1e-4 to 1e-3)
- `fp16`: Mixed precision training (saves memory)

## Single GPU Training

### Automatic GPU Detection

Simply run the script - it will automatically use GPU if available:

```bash
python examples/train_patchtst.py
```

### Force Specific GPU

```bash
# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python examples/train_patchtst.py

# Use GPU 1
CUDA_VISIBLE_DEVICES=1 python examples/train_patchtst.py
```

### Check GPU Usage

```bash
# In another terminal, monitor GPU
nvidia-smi -l 1

# Or use watch
watch -n 1 nvidia-smi
```

## Multi-GPU Training

### Using torchrun (Recommended)

```bash
# 2 GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 examples/train_patchtst.py

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 examples/train_patchtst.py

# All available GPUs
torchrun --nproc_per_node=$(nvidia-smi --list-gpus | wc -l) examples/train_patchtst.py
```

### Using General Script

```bash
torchrun --nproc_per_node=2 scripts/train.py --config_path examples/configs/patchtst.yaml
```

### Tips for Multi-GPU Training

1. **Batch Size**: Effective batch size = `batch_size * num_gpus * gradient_accumulation_steps`
2. **Learning Rate**: May need to scale with batch size
3. **Sync BatchNorm**: Automatically handled by DDP
4. **Gradient Accumulation**: Useful when GPU memory is limited

## CPU Training

The framework automatically falls back to CPU if no GPU is detected:

```bash
# Just run normally - will use CPU if no GPU
python examples/train_patchtst.py
```

**Note**: CPU training is much slower than GPU. Recommended only for:
- Testing code
- Small datasets
- Development/debugging

## Monitoring Training

### TensorBoard

Training logs are saved to TensorBoard:

```bash
# Start TensorBoard
tensorboard --logdir outputs/

# Open browser to http://localhost:6006
```

### Console Output

Training progress is logged to console:

```
2025-11-21 12:00:00 INFO: Epoch 1/10
2025-11-21 12:00:05 INFO: Train Loss: 0.1234
2025-11-21 12:00:10 INFO: Valid Loss: 0.1456 (Best: 0.1456)
2025-11-21 12:00:10 INFO: Checkpoint saved: outputs/patchtst_example/checkpoint_epoch_1.pt
```

### Checkpoints

Checkpoints are saved in the output directory:

```
outputs/patchtst_example/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── checkpoint_epoch_3.pt
└── best_model.pt
```

### Resuming Training

```yaml
# In your config file
resume:
  checkpoint_path: "outputs/patchtst_example/checkpoint_epoch_5.pt"
```

Then run training normally - it will resume from the checkpoint.

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce batch size in config:
   ```yaml
   training:
     batch_size: 16  # Try smaller values: 16, 8, 4
   ```

2. Enable mixed precision:
   ```yaml
   training:
     fp16: True
   ```

3. Use gradient accumulation:
   ```yaml
   training:
     batch_size: 8
     gradient_accumulation_steps: 4  # Effective batch size = 32
   ```

4. Reduce model size:
   ```yaml
   model:
     d_model: 64      # Smaller hidden size
     e_layers: 2      # Fewer layers
   ```

### Slow Training

**Solutions:**
1. Use GPU instead of CPU
2. Increase batch size (if memory allows)
3. Reduce number of workers if I/O bound:
   ```yaml
   training:
     num_workers: 0  # Or 1
   ```
4. Use mixed precision (fp16)

### NaN Loss

**Solutions:**
1. Reduce learning rate:
   ```yaml
   training:
     learning_rate: 0.0001  # Smaller value
   ```

2. Enable gradient clipping:
   ```yaml
   training:
     max_grad_norm: 1.0
   ```

3. Check data normalization:
   ```yaml
   data:
     common:
       normalize: True
   ```

### Data Loading Errors

**Check:**
1. Data path is correct
2. Parquet file exists
3. Data has required columns (date, value)
4. No corrupted files

**Generate test data:**
```bash
python examples/create_data.py
```

### Import Errors

**Missing dependencies:**
```bash
# Install core dependencies
pip install -r requirements.txt

# Install optional dependencies
pip install einops pyarrow
```

## Best Practices

### For Fast Training
- Use GPU with mixed precision (fp16)
- Increase batch size to maximize GPU utilization
- Use efficient models (DLinear for simple patterns)
- Enable data pinning: `pin_memory: True`

### For Best Accuracy
- Train longer (more epochs)
- Use larger models (more layers, larger hidden size)
- Enable early stopping to avoid overfitting
- Tune learning rate and weight decay
- Use validation set for hyperparameter tuning

### For Large Datasets
- Use multi-GPU training
- Enable gradient accumulation
- Use mixed precision to save memory
- Consider data sampling for faster iteration

## Next Steps

- **Evaluation**: See [EVALUATION.md](EVALUATION.md) for how to evaluate models
- **Examples**: See [EXAMPLES.md](EXAMPLES.md) for detailed script explanations
- **Configuration**: Modify YAML files in `examples/configs/` for your use case

