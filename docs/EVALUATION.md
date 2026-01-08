# Evaluation

To run evaluation with 2 gpus

```bash
python scripts/evaluate.py --config_path configs/config.yaml --num_gpus 2
```

## Chronos-2

```bash
pip install chronos-forecasting
```

Download pretrained model from huggingface

In config.yaml, set `model_name` to "ChronosV2". For local usage, set `pretrained_model_name_or_path` to 
your downloaded directory. For online, set `pretrained_model_name_or_path` to "amazon/chronos-2".


## TimesFM-2.5

following the instruction in https://github.com/google-research/timesfm/tree/master to install the repo.

In config.yaml, set `model_name` to "TimesFMV2p5". For local usage, set `pretrained_model_name_or_path` to 
your downloaded directory. For online, set `pretrained_model_name_or_path` to "google/timesfm-2.5-200m-pytorch".

## TiRex-Zero

following the instruction in https://github.com/NX-AI/tirex/tree/main to install the repo.

In config.yaml, set `model_name` to "TiRex". For local usage, set `pretrained_model_name_or_path` to 
your downloaded directory. For online, set `pretrained_model_name_or_path` to "NX-AI/TiRex".