# FINETUNE

To perform finetune on TRAIN part of TRAIN/TEST set:

```bash
torchrun --nproc_per_node 4 scripts/finetune.py --config_path configs/finetune/{model}/{config}.yaml
```