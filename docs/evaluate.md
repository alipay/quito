# Evaluation

To run evaluation with 2 processes:

## Using CLI (Recommended)
```bash
quito-cli evaluate --config_path configs/evaluate/{model}/config.yaml --num_processes 2 --use_gpu 1
```

## Using Script Directly
```bash
python quito/scripts/evaluate.py --config_path configs/evaluate/{model}/config.yaml --num_processes 2 --use_gpu 1
```

## Arguments
- `--config_path`: Path to the evaluation configuration YAML file (required)
- `--num_processes`: Number of processes to use for evaluation (default: 1)
- `--use_gpu`: Whether to use GPU for evaluation (0 or 1) (default: 1)

## Foundation Model Configuration

### Chronos-2

```bash
pip install chronos-forecasting
```

Download pretrained model from Hugging Face or use online version.

In config.yaml, set `model_name` to "ChronosV2". For local usage, set `pretrained_model_name_or_path` to 
your downloaded directory. For online, set `pretrained_model_name_or_path` to "amazon/chronos-2".

### TimesFM-2.5

Follow the instructions at https://github.com/google-research/timesfm/tree/master to install the repo.

In config.yaml, set `model_name` to "TimesFMV2p5". For local usage, set `pretrained_model_name_or_path` to 
your downloaded directory. For online, set `pretrained_model_name_or_path` to "google/timesfm-2.5-200m-pytorch".

### TiRex-Zero

Follow the instructions at https://github.com/NX-AI/tirex/tree/main to install the repo.

In config.yaml, set `model_name` to "TiRex". For local usage, set `pretrained_model_name_or_path` to 
your downloaded directory. For online, set `pretrained_model_name_or_path` to "NX-AI/TiRex".