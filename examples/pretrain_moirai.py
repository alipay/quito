#!/usr/bin/env python3
"""
Pretrain Moirai Model Example

This script demonstrates how to pretrain (or fine-tune) the Salesforce Moirai model
using the QUITO framework.

It uses the masked autoencoder objective provided by the Moirai library (uni2ts).
"""

import sys
import logging
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quito.utils.distributed import setup, DistributedGroupManager
from quito.config.auto import AutoConfig
from quito.models.base import BaseModel
from quito.datasets import get_dataset, TimeSeriesDataLoader
from quito.trainers.trainers import Trainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    # Configuration path
    config_path = Path(__file__).parent / "configs/moirai_pretrain.yaml"
    
    # Setup distributed environment (or CPU/Single GPU)
    rank, world_size, local_rank, config, output_dir = setup(str(config_path), mode="pretrain")
    
    with DistributedGroupManager(config.distributed.ddp_backend, rank, local_rank, world_size) as dist_group:
        # 1. Parse Configuration
        data_config, model_config, training_config = AutoConfig.from_config(
            config, 
            local_rank=local_rank, 
            rank=rank, 
            world_size=world_size
        )
        
        # 2. Setup Dataset & Dataloaders
        # Note: For Moirai pretraining, we use standard TimeSeriesDataset.
        # The model wrapper handles concatenating x+y and masking.
        train_ds, eval_ds, test_ds = get_dataset(data_config)
        
        if not train_ds:
            logger.error("No training dataset found!")
            return

        logger.info(f"Train dataset size: {len(train_ds)}")
        if eval_ds:
            logger.info(f"Eval dataset size: {len(eval_ds)}")
            
        # 3. Setup Model
        # This will initialize MoriaiModel in 'pretrain' mode as specified in config
        model = BaseModel.from_config(model_config, local_rank=local_rank)
        
        # 4. Setup Trainer
        trainer = Trainer(
            model=model,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            config=training_config,
            local_rank=local_rank,
            global_rank=rank,
            world_size=world_size
        )
        
        # 5. Start Training
        trainer.train()


if __name__ == "__main__":
    main()

