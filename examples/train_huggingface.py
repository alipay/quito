#!/usr/bin/env python3
"""
HuggingFace Time Series Model - Training & Evaluation

Self-contained script to train/evaluate any HuggingFace time series model.
No dependencies on other example scripts.
Automatically uses CPU if no GPU is available.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from quito.config.auto import AutoConfig
from quito.config.training import TaskType, ModeType
from quito.models.auto import AutoModel
from quito.trainers.auto import AutoTrainer
from quito.utils.distributed import setup, DistributedGroupManager
from quito.utils.common import set_seed
from quito.datasets import load_datasets


def main():
    """Main training function for HuggingFace models."""
    config_path = Path(__file__).parent / "configs/huggingface.yaml"
    
    print("=" * 80)
    print("HuggingFace Time Series Model - Training Pipeline")
    print("=" * 80)
    
    rank, world_size, local_rank, config, output_dir = setup(
        config_path_or_obj=str(config_path), mode=TaskType.PRE_TRAIN
    )
    
    data_config, model_config, training_config = AutoConfig.from_config(
        config, rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    if rank == 0:
        data_config.save(os.path.join(output_dir, 'data_config.yaml'))
        model_config.save(os.path.join(output_dir, 'model_config.yaml'))
        training_config.save(os.path.join(output_dir, 'training_config.yaml'))
    
    with DistributedGroupManager(
        backend=training_config.ddp_backend, rank=rank, 
        local_rank=local_rank, world_size=world_size
    ):
        set_seed(training_config.seed + local_rank)
        
        train_dataset = load_datasets(
            data_config=data_config, task=TaskType.PRE_TRAIN, mode=ModeType.TRAIN
        )
        valid_dataset = load_datasets(
            data_config=data_config, task=TaskType.PRE_TRAIN, mode=ModeType.VALID
        )
        
        model = AutoModel.from_config(config=model_config, local_rank=local_rank if torch.cuda.is_available() else -1)
        
        # Handle device placement
        device = "cpu"
        if torch.cuda.is_available() and world_size > 1:
            device = f"cuda:{local_rank}"
            model = model.to(device)
            model = DDP(model, device_ids=[local_rank],
                       find_unused_parameters=training_config.ddp_find_unused_parameters)
            logging.info(f"Using GPU {local_rank} with DDP")
        elif torch.cuda.is_available():
            device = "cuda:0"
            model = model.to(device)
            logging.info("Using single GPU")
        else:
            model = model.to(device)
            logging.info("Using CPU")
        
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
        
        trainer = AutoTrainer.from_config(
            model=model, train_dataset=train_dataset, eval_dataset=valid_dataset,
            config=training_config, local_rank=trainer_local_rank,
            global_rank=trainer_global_rank, world_size=trainer_world_size
        )
        
        try:
            results = trainer.train()
            logging.info(f"✅ Completed! {results}")
        except Exception as e:
            logging.error(f"❌ Failed: {e}")
            raise


if __name__ == "__main__":
    main()
