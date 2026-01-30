# Hyperparameter Tuning

To perform hyperparameter tuning using TRAIN/VALID part of TRAIN/TEST set:

## Using CLI (Recommended)
```bash
quito-cli tune --config_path configs/tune/{model}/config.yaml \
              --tuning_config_path configs/tune/{model}/tune_config.yaml \
              --num_processes 4 \
              --num_samples 100 \
              --use_gpu 1
```

## Using Script Directly
```bash
python quito/scripts/tune.py \
--config_path configs/tune/{model}/config.yaml \
--tuning_config_path configs/tune/{model}/tune_config.yaml \
--num_processes 4 \
--num_samples 100 \
--use_gpu 1
```

## Arguments
- `--config_path`: Path to the base configuration YAML file (required)
- `--tuning_config_path`: Path to the tuning configuration YAML file (required)
- `--num_processes`: Number of parallel workers for hyperparameter search (default: 1)
- `--num_samples`: Number of hyperparameter samples to try (default: 10)
- `--use_gpu`: Whether to use GPU for tuning (0 or 1) (default: 1)