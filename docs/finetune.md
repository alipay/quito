# Fine-tuning

To perform fine-tuning on the TRAIN part of TRAIN/TEST set:

## Using CLI (Recommended)
```bash
quito-cli finetune --config_path configs/finetune/{model}/config.yaml --num_processes 4 --use_gpu 1
```

## Using Script Directly
```bash
torchrun --nproc_per_node 4 quito/scripts/finetune.py --config_path configs/finetune/{model}/config.yaml --use_gpu 1
```

## Arguments
- `--config_path`: Path to the fine-tuning configuration YAML file (required)
- `--num_processes`: Number of processes for distributed training (default: 6)
- `--use_gpu`: Whether to use GPU for training (0 or 1) (default: 1)
- `--seed`: Random seed for reproducibility (optional)