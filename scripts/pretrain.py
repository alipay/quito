
#!/usr/bin/env python3
"""
Training script for time series forecasting models.

This script uses YAML configuration files for all training parameters.

Usage:
    python scripts/train.py --config configs/train_config.yaml
    python scripts/train.py --config configs/pyraformer_gpu.yaml
"""
import os
import argparse
import logging
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf, DictConfig
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from quito.config.auto import AutoConfig
from quito.config.training import TaskType, ModeType
from quito.models.auto import AutoModel
from quito.trainers.auto import AutoTrainer
from quito.utils.distributed import setup, DistributedGroupManager
from quito.utils.common import set_seed
from quito.datasets import load_datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train time series forecasting models using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default config
  python scripts/train.py --config configs/train_config.yaml
  
  # Train with GPU-optimized config
  python scripts/train.py --config configs/pyraformer_gpu.yaml
  
  # Quick test training
  python scripts/train.py --config configs/informer_quick.yaml
        """
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML config file (required)"
    )
    
    return parser.parse_args()

def main():
    """Main training function."""
    args = parse_args()
    rank, world_size, local_rank, config, output_dir = setup(config_path=args.config_path, mode=TaskType.PRE_TRAIN)
    # load config
    data_config, model_config, training_config = AutoConfig.from_config(config, rank=rank, world_size=world_size, local_rank=local_rank)
    # 
    # save config
    if rank == 0:
        data_config.save(os.path.join(output_dir, 'data_config.yaml'))
        model_config.save(os.path.join(output_dir, 'model_config.yaml'))
        training_config.save(os.path.join(output_dir, 'training_config.yaml'))
    
    with DistributedGroupManager(backend=training_config.ddp_backend, rank=rank, local_rank=local_rank, world_size=world_size) as group_manager:
        # Set up distributed training if available
        # Set seed, each process get a different seed
        set_seed(training_config.seed + local_rank)
        # Init training dataset
        train_dataset = load_datasets(
            data_config=data_config,
            task=TaskType.PRE_TRAIN,
            mode=ModeType.TRAIN
        )    
        valid_dataset = load_datasets(
            data_config=data_config,
            task=TaskType.PRE_TRAIN,
            mode=ModeType.VALID
        )
        
        logging.info(f"Training dataset size: {len(train_dataset)} samples")
        logging.info(f"Validation dataset size: {len(valid_dataset)} samples")

        # Create model
        model = AutoModel.from_config(config=model_config, local_rank=local_rank)
        logging.info(f"Model created: {model.__class__.__name__}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        model = model.to(local_rank)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=training_config.ddp_find_unused_parameters)
    
        # Create trainer
        trainer = AutoTrainer.from_config(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            config=training_config,
            local_rank=local_rank,
            global_rank=rank,
            world_size=world_size
        )
        logging.info(f"Trainer {trainer.__class__.__name__} created ...")
        
        try:
            results = trainer.train()
    
            logging.info("=" * 80)
            logging.info("Training completed successfully!")
            logging.info('The results are: \n')
            logging.info(results)
        
        except Exception as e:
            logging.error(f"Training failed with error: {e}, peform cleaning ...")
            raise
        

if __name__ == "__main__":
    main()
