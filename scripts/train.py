#!/usr/bin/env python3
"""
General Training Script for Time Series Models

This is the main training script that can be used with any model configuration.
For specific model examples, see the examples/ folder.

Usage:
    python scripts/train.py --config_path configs/config.yaml
    python scripts/train.py --config_path examples/configs/chronos.yaml
"""
import os
import argparse
import logging
import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

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
  python scripts/train.py --config_path configs/config.yaml
  
  # Train Chronos model
  python scripts/train.py --config_path examples/configs/chronos.yaml
  
  # Or use dedicated example scripts
  python examples/train_chronos.py
  python examples/train_moriai.py
        """
    )
    
    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML config file (required)"
    )
    
    return parser.parse_args()


def main(config=None):
    """Main training function."""
    if config is None:
        args = parse_args()
        config_source = args.config_path
    else:
        config_source = config

    rank, world_size, local_rank, config, output_dir = setup(
        config_path_or_obj=config_source, 
        mode=TaskType.PRE_TRAIN
    )
    
    # load config
    data_config, model_config, training_config = AutoConfig.from_config(
        config, 
        rank=rank, 
        world_size=world_size, 
        local_rank=local_rank
    )
    
    # save config
    if rank == 0:
        data_config.save(os.path.join(output_dir, 'data_config.yaml'))
        model_config.save(os.path.join(output_dir, 'model_config.yaml'))
        training_config.save(os.path.join(output_dir, 'training_config.yaml'))
    
    with DistributedGroupManager(
        backend=training_config.ddp_backend, 
        rank=rank, 
        local_rank=local_rank, 
        world_size=world_size
    ) as group_manager:
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
        model = AutoModel.from_config(
            config=model_config, 
            local_rank=local_rank if torch.cuda.is_available() else -1
        )
        logging.info(f"Model created: {model.__class__.__name__}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Handle device placement - default to CPU if no GPU
        device = "cpu"
        if torch.cuda.is_available() and world_size > 1:
            device = f"cuda:{local_rank}"
            model = model.to(device)
            model = DDP(
                model, 
                device_ids=[local_rank], 
                find_unused_parameters=training_config.ddp_find_unused_parameters
            )
            logging.info(f"Using GPU {local_rank} with DDP")
        elif torch.cuda.is_available():
            device = "cuda:0"
            model = model.to(device)
            logging.info("Using single GPU")
        else:
            model = model.to(device)
            logging.info("Using CPU (no GPU detected)")

    # Set ranks for trainer based on device availability
    if torch.cuda.is_available():
        trainer_local_rank = local_rank
        trainer_global_rank = rank
        trainer_world_size = world_size
    else:
        # CPU mode: set all to -1
        trainer_local_rank = -1
        trainer_global_rank = -1
        trainer_world_size = -1
    
    # Create trainer
    trainer = AutoTrainer.from_config(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        config=training_config,
        local_rank=trainer_local_rank,
        global_rank=trainer_global_rank,
        world_size=trainer_world_size
    )
        logging.info(f"Trainer {trainer.__class__.__name__} created ...")
        
        try:
            results = trainer.train()
    
            logging.info("=" * 80)
            logging.info("Training completed successfully!")
            logging.info('The results are: \n')
            logging.info(results)
        
        except Exception as e:
            logging.error(f"Training failed with error: {e}, perform cleaning ...")
            raise
        

if __name__ == "__main__":
    main()

