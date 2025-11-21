# Examples Guide

Detailed explanation of all example scripts in the `examples/` folder.

## Table of Contents

- [Overview](#overview)
- [Training Examples](#training-examples)
- [Evaluation Examples](#evaluation-examples)
- [Utility Scripts](#utility-scripts)
- [Configuration Files](#configuration-files)

## Overview

The `examples/` folder contains self-contained scripts that demonstrate how to use QUITO:

```
examples/
├── train_patchtst.py        # Train PatchTST model
├── train_huggingface.py     # Train HuggingFace models
├── eval_chronos.py          # Evaluate Chronos (zero-shot)
├── eval_moriai.py           # Evaluate Moirai (zero-shot)
├── create_data.py           # Generate dummy data
├── configs/                 # Configuration files
│   ├── patchtst.yaml
│   ├── huggingface.yaml
│   ├── chronos.yaml
│   └── moriai.yaml
└── datasets/                # Data storage
    └── parquet_data/
```

**Key Features:**
- ✅ **Self-contained** - No command-line arguments needed
- ✅ **Independent** - No cross-imports between examples
- ✅ **Educational** - Clear, well-commented code
- ✅ **Production-ready** - Can be adapted for real use

## Training Examples

### train_patchtst.py

**Purpose:** Train a PatchTST model on your time series data.

**Usage:**
```bash
python examples/train_patchtst.py
```

**What it does:**
1. Loads configuration from `examples/configs/patchtst.yaml`
2. Sets up distributed training environment (if multiple GPUs)
3. Loads training and validation datasets
4. Creates PatchTST model
5. Initializes trainer
6. Trains the model with checkpointing and early stopping
7. Saves best model

**Configuration:** `examples/configs/patchtst.yaml`

**Key Code Sections:**

```python
# 1. Setup distributed training
rank, world_size, local_rank, config, output_dir = setup(
    config_path_or_obj=str(config_path), 
    mode=TaskType.PRE_TRAIN
)

# 2. Load datasets
train_dataset = load_datasets(
    data_config=data_config,
    task=TaskType.PRE_TRAIN,
    mode=ModeType.TRAIN
)

# 3. Create model
model = AutoModel.from_config(
    config=model_config,
    local_rank=local_rank
)

# 4. Handle device placement
if torch.cuda.is_available() and world_size > 1:
    model = DDP(model, device_ids=[local_rank])

# 5. Create trainer and train
trainer = AutoTrainer.from_config(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    config=training_config,
    ...
)
results = trainer.train()
```

**Customization:**

To modify for your needs:
1. Edit `examples/configs/patchtst.yaml` for hyperparameters
2. Change data path in config
3. Adjust model architecture in config
4. Modify logging/checkpointing settings

**Output:**
```
outputs/patchtst_example/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── best_model.pt
├── config.yaml
└── logs/
```

---

### train_huggingface.py

**Purpose:** Train any HuggingFace time series model.

**Usage:**
```bash
python examples/train_huggingface.py
```

**What it does:**
- Similar to `train_patchtst.py` but uses HuggingFace models
- Supports any HF time series model (TimesFM, etc.)
- Leverages HuggingFace's pre-trained weights

**Configuration:** `examples/configs/huggingface.yaml`

**Key Features:**
- Auto-downloads models from HuggingFace Hub
- Supports `trust_remote_code` for custom models
- Compatible with HF's model ecosystem

**Example Models:**
- `google/timesfm-1.0-200m` - Google's TimesFM
- Any HuggingFace time series model

**Customization:**

Change model in config:
```yaml
model:
  model_name: "huggingface"
  pretrained_model_name_or_path: "google/timesfm-1.0-200m"
  trust_remote_code: True
```

---

## Evaluation Examples

### eval_chronos.py

**Purpose:** Zero-shot evaluation using Amazon Chronos pre-trained model.

**Usage:**
```bash
python examples/eval_chronos.py
```

**What it does:**
1. Loads pre-trained Chronos model from HuggingFace
2. Loads test dataset
3. Generates predictions without any training
4. Computes evaluation metrics (MSE, MAE)
5. Reports results

**Configuration:** `examples/configs/chronos.yaml`

**Key Code Sections:**

```python
# 1. Load pre-trained model
model = AutoModel.from_config(
    config=model_config,
    local_rank=local_rank
)
model.eval()

# 2. Evaluate
with torch.no_grad():
    for batch in test_loader:
        x, y = batch['x'], batch['y']
        y_pred = model.predict(x)
        
        all_predictions.append(y_pred)
        all_targets.append(y[:, -horizon:, :])

# 3. Compute metrics
mse = F.mse_loss(all_predictions, all_targets)
mae = F.l1_loss(all_predictions, all_targets)
```

**When to use:**
- Quick baseline without training
- Comparing against SOTA pre-trained models
- Limited training data available
- Prototyping phase

**Available Chronos Models:**
```yaml
# Smallest (fastest)
pretrained_model_name_or_path: "amazon/chronos-t5-tiny"

# Small (balanced)
pretrained_model_name_or_path: "amazon/chronos-t5-small"

# Large (most accurate)
pretrained_model_name_or_path: "amazon/chronos-t5-large"
```

---

### eval_moriai.py

**Purpose:** Zero-shot evaluation using Salesforce Moirai pre-trained model.

**Usage:**
```bash
python examples/eval_moriai.py
```

**What it does:**
- Similar to `eval_chronos.py` but uses Moirai model
- Supports multi-variate time series
- No training required

**Configuration:** `examples/configs/moriai.yaml`

**Key Parameters:**
```yaml
model:
  prediction_length: 96        # Forecast horizon
  context_length: 512          # Input context length
  patch_size: 64               # Patch size for model
  num_samples: 100             # Number of samples for prediction
```

**When to use:**
- Multi-variate time series
- Need probabilistic forecasts
- Comparing against Salesforce's SOTA model

---

## Utility Scripts

### create_data.py

**Purpose:** Generate dummy time series data for testing.

**Usage:**
```bash
python examples/create_data.py
```

**What it does:**
1. Creates synthetic time series data
2. Saves as Parquet files
3. Places in `examples/datasets/parquet_data/`

**Output:**
```
examples/datasets/parquet_data/
└── open_hour_train/
    └── hour_train_hour_p1.parquet
```

**Generated Data Format:**
- Date column: datetime index
- Value column: synthetic time series values
- Patterns: trends, seasonality, noise

**Customization:**

Edit the script to change:
- Number of time series
- Length of each series
- Patterns (trend, seasonality, noise level)
- Frequency (hourly, daily, etc.)

**Use Cases:**
- Testing the pipeline without real data
- Development and debugging
- Learning how the framework works
- Quick prototyping

---

## Configuration Files

Located in `examples/configs/`, these YAML files control all aspects of training/evaluation.

### patchtst.yaml

**Model:** PatchTST transformer

**Key Settings:**
```yaml
model:
  model_name: "patchtst"
  patch_len: 16                # Length of each patch
  stride: 8                    # Stride between patches
  d_model: 128                 # Hidden dimension
  n_heads: 8                   # Number of attention heads
  e_layers: 3                  # Number of layers
```

**Best For:**
- General-purpose time series
- Long sequences
- Complex patterns

---

### huggingface.yaml

**Model:** Generic HuggingFace wrapper

**Key Settings:**
```yaml
model:
  model_name: "huggingface"
  pretrained_model_name_or_path: "google/timesfm-1.0-200m"
  trust_remote_code: True
```

**Best For:**
- Using specific HF models
- Leveraging pre-trained weights
- Custom HF architectures

---

### chronos.yaml

**Model:** Amazon Chronos (inference only)

**Key Settings:**
```yaml
model:
  model_name: "chronos"
  pretrained_model_name_or_path: "amazon/chronos-t5-small"
  num_samples: 20              # Number of forecast samples
  temperature: 1.0             # Sampling temperature
  top_k: 50                    # Top-k sampling
  top_p: 1.0                   # Nucleus sampling
```

**Best For:**
- Zero-shot baselines
- Quick evaluation
- Comparing against SOTA

---

### moriai.yaml

**Model:** Salesforce Moirai (inference only)

**Key Settings:**
```yaml
model:
  model_name: "moriai"
  pretrained_model_name_or_path: "Salesforce/moirai-1.0-R-small"
  prediction_length: 96
  context_length: 512
  patch_size: 64
  num_samples: 100
```

**Best For:**
- Multi-variate forecasting
- Probabilistic predictions
- Zero-shot evaluation

---

## Common Patterns

### Pattern 1: Quick Testing

```bash
# 1. Generate test data
python examples/create_data.py

# 2. Test training
python examples/train_patchtst.py

# 3. Test evaluation
python examples/eval_chronos.py
```

### Pattern 2: Model Comparison

```bash
# Train your model
python examples/train_patchtst.py > results_patchtst.txt

# Compare with baselines
python examples/eval_chronos.py > results_chronos.txt
python examples/eval_moriai.py > results_moriai.txt

# Analyze results
cat results_*.txt | grep "MSE:"
```

### Pattern 3: Hyperparameter Tuning

```bash
# Copy config
cp examples/configs/patchtst.yaml examples/configs/patchtst_tuned.yaml

# Edit hyperparameters in patchtst_tuned.yaml

# Modify script to use new config
# python examples/train_patchtst.py  (edit config_path in script)
```

### Pattern 4: Production Deployment

1. Train on full dataset
2. Evaluate on held-out test set
3. Save best checkpoint
4. Create inference script (similar to eval scripts)
5. Deploy model

---

## Advanced Usage

### Custom Data Loader

Modify the dataset loading in any example script:

```python
# In train_patchtst.py
train_dataset = load_datasets(
    data_config=data_config,
    task=TaskType.PRE_TRAIN,
    mode=ModeType.TRAIN
)

# Add custom preprocessing
train_dataset = CustomDataset(train_dataset, your_transforms)
```

### Custom Model

Add your own model to the framework:

1. Create `quito/models/your_model.py`
2. Inherit from `TimeSeriesModel`
3. Register in `quito/models/auto.py`
4. Create config in `examples/configs/your_model.yaml`
5. Copy and modify an example script

### Custom Metrics

Add custom evaluation metrics:

```python
# In eval_chronos.py

def custom_metric(pred, target):
    # Your metric logic
    error = pred - target
    score = torch.mean(torch.abs(error) / (torch.abs(target) + 1e-8))
    return score.item()

metrics = {
    'mse': mse_metric,
    'mae': mae_metric,
    'mape': custom_metric,  # Custom metric
}
```

---

## Next Steps

- **Training Guide**: See [TRAINING.md](TRAINING.md) for detailed training instructions
- **Evaluation Guide**: See [EVALUATION.md](EVALUATION.md) for evaluation best practices
- **Customize**: Copy and modify example scripts for your needs
- **Contribute**: Share your custom scripts with the community!

