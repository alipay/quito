# Hyperparameter Tuning

To perform hyperparameter tuning using TRAIN VALID part of TRAIN/TEST set:

```bash
python scripts/tune.py \
--config_path configs/tune/{model}/{config}.yaml \
--tuning_config_path configs/tune/{model}/{tune_config_path}.yaml\
--num_workers 4
--num_samples 100
--use_gpu 1
```