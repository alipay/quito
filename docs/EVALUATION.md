# Evaluation Guide

Complete guide to evaluating time series models with QUITO.

## Table of Contents

- [Overview](#overview)
- [Zero-Shot Inference](#zero-shot-inference)
- [Evaluation Scripts](#evaluation-scripts)
- [Metrics](#metrics)
- [Comparing Models](#comparing-models)
- [Custom Evaluation](#custom-evaluation)

## Overview

QUITO supports two types of evaluation:

1. **Zero-Shot Inference** - Using pre-trained foundation models (Chronos, Moirai)
2. **Trained Model Evaluation** - Evaluating your trained models (PatchTST, DLinear, etc.)

## Zero-Shot Inference

Pre-trained foundation models can make predictions on your data without any training.

### Supported Models

| Model | Provider | Best For |
|-------|----------|----------|
| **Chronos** | Amazon | General time series, zero-shot baselines |
| **Moirai** | Salesforce | Multi-variate, complex patterns |

### Chronos Evaluation

```bash
# Evaluate Chronos on your test data
python examples/eval_chronos.py
```

**What it does:**
1. Loads pre-trained Chronos model from HuggingFace
2. Loads your test dataset
3. Generates predictions
4. Computes metrics (MSE, MAE)
5. Outputs results

**Features:**
- Zero-shot forecasting (pretrained on massive datasets)
- **Probabilistic forecasting** (generates samples and quantiles)
- Fine-tuning supported (use `train.py` with Chronos config)

**Configuration:** `examples/configs/chronos.yaml`

```yaml
model:
  model_name: "chronos"
  pretrained_model_name_or_path: "amazon/chronos-t5-small"
  num_samples: 20
  temperature: 1.0
  top_k: 50
  top_p: 1.0
```

**Available Models:**
- `amazon/chronos-t5-tiny` (8M params, fastest)
- `amazon/chronos-t5-mini` (20M params)
- `amazon/chronos-t5-small` (46M params, balanced)
- `amazon/chronos-t5-base` (200M params)
- `amazon/chronos-t5-large` (710M params, best accuracy)

**Probabilistic Forecasting:**
Chronos can generate full distributions. The `ChronosModel` wrapper provides:
- `predict(x)`: Returns median (point forecast)
- `predict_prob(x, quantiles=[0.1, 0.5, 0.9])`: Returns samples, quantiles, and mean

### Moirai Evaluation

```bash
# Evaluate Moirai on your test data
python examples/eval_moriai.py
```

**Configuration:** `examples/configs/moriai.yaml`

```yaml
model:
  model_name: "moriai"
  pretrained_model_name_or_path: "Salesforce/moirai-1.0-R-small"
  prediction_length: 96
  patch_size: 64
  context_length: 512
  num_samples: 100
```

**Available Models:**
- `Salesforce/moirai-1.0-R-small` (14M params)
- `Salesforce/moirai-1.0-R-base` (91M params)
- `Salesforce/moirai-1.0-R-large` (311M params)

### When to Use Zero-Shot Models

**Use Chronos/Moirai when:**
- ✅ You want quick baselines without training
- ✅ You have limited training data
- ✅ You want to compare against SOTA models
- ✅ You need fast prototyping

**Don't use when:**
- ❌ You have lots of domain-specific training data
- ❌ You need to fine-tune on your patterns
- ❌ You need the fastest inference (use DLinear)

## Evaluation Scripts

### Example Scripts

```bash
# Zero-shot inference
python examples/eval_chronos.py   # Chronos evaluation
python examples/eval_moriai.py    # Moirai evaluation
```

### Evaluation Output

```
================================================================================
Evaluation Results:
================================================================================
MSE: 0.123456
MAE: 0.234567
================================================================================
✅ Zero-shot evaluation completed successfully!
Total samples evaluated: 1000
================================================================================
```

## Metrics

### Supported Metrics

QUITO computes the following metrics by default:

**MSE (Mean Squared Error)**
- Range: [0, ∞)
- Lower is better
- Penalizes large errors heavily
- Most common for time series

**MAE (Mean Absolute Error)**
- Range: [0, ∞)
- Lower is better
- Linear penalty for errors
- More robust to outliers

### Custom Metrics

You can add custom metrics in the evaluation scripts:

```python
def custom_metric(pred, target):
    # Your metric calculation
    return score

metrics = {
    'mse': mse_metric,
    'mae': mae_metric,
    'custom': custom_metric,
}
```

### Metric Configuration

Specify which metrics to compute in your config:

```yaml
evaluation:
  eval_metrics: ['mse', 'mae', 'rmse', 'mape']
  eval_epochs: 1
  eval_steps: null
```

## Comparing Models

### Running Multiple Models

```bash
# Evaluate different models
python examples/eval_chronos.py > results_chronos.txt
python examples/eval_moriai.py > results_moriai.txt
python examples/train_patchtst.py  # Train and evaluate
```

### Example Comparison

| Model | MSE | MAE | Training Time | Inference Speed |
|-------|-----|-----|---------------|-----------------|
| Chronos-Small | 0.123 | 0.234 | 0 (pre-trained) | 50 samples/sec |
| Moirai-Small | 0.115 | 0.221 | 0 (pre-trained) | 30 samples/sec |
| PatchTST | 0.098 | 0.198 | 2 hours | 200 samples/sec |
| DLinear | 0.105 | 0.205 | 15 minutes | 500 samples/sec |

### Interpretation

**Lower MSE/MAE = Better predictions**

- Chronos/Moirai: Good zero-shot performance, no training needed
- PatchTST: Best accuracy after training on your data
- DLinear: Fast training, good accuracy for simple patterns

## Custom Evaluation

### Modify Evaluation Script

You can customize `examples/eval_chronos.py` for your needs:

```python
def main():
    # ... load model and data ...
    
    # Custom evaluation loop
    all_predictions = []
    all_targets = []
    
    for batch in test_loader:
        x, y = batch['x'], batch['y']
        
        # Get predictions
        y_pred = model.predict(x)
        
        # Custom processing
        y_pred = custom_postprocess(y_pred)
        
        all_predictions.append(y_pred)
        all_targets.append(y[:, -horizon:, :])
    
    # Compute custom metrics
    results = compute_custom_metrics(all_predictions, all_targets)
    print(results)
```

### Save Predictions

```python
# Save predictions for analysis
predictions_dict = {
    'predictions': all_predictions.numpy(),
    'targets': all_targets.numpy(),
    'timestamps': timestamps
}

import pickle
with open('predictions.pkl', 'wb') as f:
    pickle.dump(predictions_dict, f)
```

### Visualize Results

```python
import matplotlib.pyplot as plt

# Plot first sample
plt.figure(figsize=(12, 4))
plt.plot(y_true[0, :, 0], label='True', marker='o')
plt.plot(y_pred[0, :, 0], label='Predicted', marker='x')
plt.legend()
plt.title('Time Series Prediction')
plt.savefig('prediction_plot.png')
```

## Evaluation Best Practices

### For Reliable Metrics

1. **Use separate test set** - Never evaluate on training data
2. **Sufficient test samples** - At least 100+ samples
3. **Multiple runs** - Average over multiple random seeds
4. **Confidence intervals** - Report std dev or confidence intervals

### For Fair Comparison

1. **Same test set** - All models evaluated on identical data
2. **Same metrics** - Use consistent metric definitions
3. **Same preprocessing** - Identical normalization/scaling
4. **Document settings** - Record all hyperparameters

### For Production

1. **Monitor multiple metrics** - MSE, MAE, and domain-specific metrics
2. **Track over time** - Monitor model degradation
3. **A/B testing** - Compare new vs old models on live data
4. **Business metrics** - Connect to actual business impact

## Troubleshooting

### Low Accuracy

**Check:**
1. Data quality - Missing values, outliers, normalization
2. Model capacity - May need larger model
3. Hyperparameters - Learning rate, batch size
4. Training length - May need more epochs
5. Data leakage - Ensure proper train/val/test split

### Slow Evaluation

**Solutions:**
1. Increase batch size (if memory allows)
2. Use GPU instead of CPU
3. Reduce number of samples for Chronos/Moirai
4. Use smaller pre-trained model variant

### Memory Issues

**Solutions:**
1. Reduce batch size
2. Process in smaller chunks
3. Use gradient checkpointing
4. Clear cache between batches

## Next Steps

- **Training**: See [TRAINING.md](TRAINING.md) to train your own models
- **Examples**: See [EXAMPLES.md](EXAMPLES.md) for detailed script explanations
- **Configuration**: Modify YAML files in `examples/configs/` for your use case

