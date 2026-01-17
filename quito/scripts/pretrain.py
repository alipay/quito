#!/usr/bin/env python3
"""
Pre-training script for time series forecasting models.

This script handles pre-training models from scratch using distributed training
with torchrun. It uses YAML configuration files for all training parameters.

Usage:
    torchrun --nproc_per_node=4 quito/scripts/pretrain.py \\
        --config_path configs/pretrain/patchtst/config.yaml \\
        --use_gpu=1
"""
import os
import argparse
import logging
import sys
from pathlib import Path
import torch
import torch.distributed as dist

from quito.config.auto import AutoConfig
from quito.config.training import TaskType, ModeType
from quito.models.auto import AutoModel
from quito.trainers.auto import AutoTrainer
from quito.utils.distributed import setup_train, setup_logging
from quito.utils.common import set_seed
from quito.datasets import load_datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def parse_args():
    """
    Parse command line arguments for pre-training.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pre-train time series forecasting models using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    torchrun --nproc_per_node=4 quito/scripts/pretrain.py \\
        --config_path configs/pretrain/patchtst/config.yaml \\
        --use_gpu=1
        """
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML config file (required)"
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        help="Whether to use GPU (0 or 1)"
    )
    return parser.parse_args()


def main():
    """
    Main pre-training function.
    
    Orchestrates the pre-training process:
    1. Parse command line arguments
    2. Set up distributed training environment
    3. Load configuration
    4. Initialize datasets
    5. Create model and trainer
    6. Run training
    
    Raises:
        RuntimeError: If training fails.
    """
    args = parse_args()
    
    # Set up distributed training
    rank, world_size, local_rank, config, output_dir = setup_train(
        args.config_path, 
        mode=TaskType.PRE_TRAIN
    )
    
    # Set random seed
    set_seed(config.training.seed + local_rank)
    
    # Save configs
    if rank == 0:
        data_config, model_config, training_config = AutoConfig.from_config(
            config, rank=rank, world_size=world_size, local_rank=local_rank
        )
        data_config.save(os.path.join(output_dir, 'data_config.yaml'))
        model_config.save(os.path.join(output_dir, 'model_config.yaml'))
        training_config.save(os.path.join(output_dir, 'training_config.yaml'))
    else:
        data_config, model_config, training_config = AutoConfig.from_config(
            config, rank=rank, world_size=world_size, local_rank=local_rank
        )
    
    # Load datasets
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
    
    if rank == 0:
        logging.info(f"Training dataset size: {len(train_dataset)} samples")
        logging.info(f"Validation dataset size: {len(valid_dataset)} samples")
    
    # Create model
    model = AutoModel.from_config(config=model_config, local_rank=local_rank)
    if rank == 0:
        logging.info(f"Model created: {model.__class__.__name__}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = AutoTrainer.from_config(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        config=training_config,
        local_rank=local_rank,
        global_rank=rank,
        world_size=world_size,
        use_gpu=args.use_gpu,
    )
    if rank == 0:
        logging.info(f"Trainer {trainer.__class__.__name__} created ...")
    
    # Train
    try:
        results = trainer.train()
        if rank == 0:
            logging.info("=" * 80)
            logging.info("Pre-training completed successfully!")
            logging.info('The results are: \n')
            logging.info(results)
    except Exception as e:
        logging.error(f"Pre-training failed: {e}")
        raise


if __name__ == "__main__":
    main()
