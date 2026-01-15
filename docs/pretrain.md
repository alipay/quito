# Pre-training

To perform pre-training (train on pre-train dataset):
```bash
torchrun --nproc_per_node 4 scripts/pretrain.py --config_path configs/pretrain/{model}/{config}.yaml
```
