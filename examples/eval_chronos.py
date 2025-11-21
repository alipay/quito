#!/usr/bin/env python3
"""
Chronos Time Series Model - Zero-Shot Inference & Evaluation

Self-contained script to evaluate Amazon Chronos model on your data.
⚠️  NOTE: Chronos is a pre-trained zero-shot model for INFERENCE ONLY.
While it technically supports fine-tuning, it's primarily designed for zero-shot forecasting.

This script demonstrates:
- Loading pre-trained Chronos model from HuggingFace
- Evaluating on custom time series data
- Computing metrics (MSE, MAE, etc.)

No dependencies on other example scripts.
Automatically uses CPU if no GPU is available.

DEPENDENCIES:
  Core: requirements.txt
  Chronos (required): pip install git+https://github.com/amazon-science/chronos-forecasting.git
  See requirements-optional.txt for details
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import torch
from omegaconf import OmegaConf

from quito.models.auto import AutoModel
from quito.config.auto import AutoConfig
from quito.datasets import load_datasets
from quito.config.training import TaskType, ModeType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s'
)


def main():
    """Main evaluation function for Chronos zero-shot inference."""
    
    # Load configuration
    config_path = Path(__file__).parent / "configs" / "chronos.yaml"
    cfg = OmegaConf.load(config_path)
    
    logging.info("="*80)
    logging.info("Chronos Zero-Shot Inference & Evaluation")
    logging.info("="*80)
    
    # Simple setup for inference (no distributed training needed)
    rank = 0
    local_rank = 0 if torch.cuda.is_available() else -1
    world_size = 1
    
    # Parse configs - pass the full config to AutoConfig
    data_config, model_config, training_config = AutoConfig.from_config(
        cfg, rank=rank, world_size=world_size, local_rank=local_rank
    )
    
    logging.info("Loading datasets...")
    # Load test dataset only (no training)
    test_dataset = load_datasets(
        data_config=data_config,
        task=TaskType.PRE_TRAIN,
        mode=ModeType.TEST
    )
    
    logging.info(f"Test samples: {len(test_dataset)}")
    
    logging.info("Loading pre-trained Chronos model...")
    model = AutoModel.from_config(
        config=model_config, 
        local_rank=local_rank if torch.cuda.is_available() else -1
    )
    logging.info(f"Model: {model.__class__.__name__}")
    logging.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Handle device placement
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        model = model.to(device)
        logging.info("Using single GPU")
    else:
        model = model.to(device)
        logging.info("Using CPU")
    
    # Set to evaluation mode
    model.eval()
    
    # Create dataloader for evaluation
    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
    )
    
    # Setup metrics - use torch functions directly
    import torch.nn.functional as F
    
    def mse_metric(pred, target):
        return F.mse_loss(pred, target).item()
    
    def mae_metric(pred, target):
        return F.l1_loss(pred, target).item()
    
    metrics = {
        'mse': mse_metric,
        'mae': mae_metric,
    }
    
    logging.info("="*80)
    logging.info("Starting zero-shot evaluation...")
    logging.info("="*80)
    
    # Evaluate
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            x = batch['x']
            y = batch['y']
            
            # Get predictions
            try:
                y_pred = model.predict(x)
                
                # Extract target horizon
                y_true = y[:, -model.forecast_horizon:, :]
                
                all_predictions.append(y_pred.cpu())
                all_targets.append(y_true.cpu())
                
                if (batch_idx + 1) % 10 == 0:
                    logging.info(f"Processed {batch_idx + 1}/{len(test_loader)} batches")
                    
            except Exception as e:
                logging.error(f"Error processing batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Compute metrics
    if len(all_predictions) > 0:
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        logging.info("="*80)
        logging.info("Evaluation Results:")
        logging.info("="*80)
        
        for metric_name, metric_fn in metrics.items():
            score = metric_fn(all_predictions, all_targets)
            logging.info(f"{metric_name.upper()}: {score:.6f}")
        
        logging.info("="*80)
        logging.info("✅ Zero-shot evaluation completed successfully!")
        logging.info(f"Total samples evaluated: {len(all_predictions)}")
        logging.info("="*80)
    else:
        logging.error("❌ No predictions generated. Check for errors above.")


if __name__ == "__main__":
    main()

