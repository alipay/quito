# Pre-training

To perform pre-training (train on pre-train dataset):

## Using CLI (Recommended)
```bash
quito-cli pretrain --config_path configs/pretrain/{model}/config.yaml --num_processes 4 --use_gpu 1
```

## Using Script Directly
```bash
torchrun --nproc_per_node 4 quito/scripts/pretrain.py --config_path configs/pretrain/{model}/config.yaml --use_gpu 1
```

## Arguments
- `--config_path`: Path to the training configuration YAML file (required)
- `--num_processes`: Number of processes for distributed training (default: 6)
- `--use_gpu`: Whether to use GPU for training (0 or 1) (default: 1)
