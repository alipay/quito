#!/usr/bin/env python3
"""
Evaluation script for time series forecasting models.

This script evaluates trained models on test data using Ray for distributed
evaluation across multiple GPUs. It uses YAML configuration files for all
evaluation parameters.

Usage:
    python quito/scripts/evaluate.py \\
        --config_path configs/evaluate/patchtst/config.yaml \\
        --num_gpus 2
"""
import os
import argparse
import logging
import sys
from pathlib import Path
import torch
import ray
from typing import Dict, List

from quito.config.auto import AutoConfig
from quito.config.training import TaskType, ModeType
from quito.models.auto import AutoModel
from quito.trainers.auto import AutoTrainer
from quito.utils.distributed import setup_evaluation
from quito.utils.common import set_seed
from quito.datasets import load_datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@ray.remote(num_gpus=1)
class ModelEvaluator:
    """
    Ray actor for distributed model evaluation.
    
    Each actor evaluates the model on a subset of test data using a single GPU.
    This enables parallel evaluation across multiple GPUs.
    
    Attributes:
        config: Model and data configuration
        gpu_id: GPU ID assigned to this actor
    """
    
    def __init__(self, config, gpu_id: int):
        """
        Initialize the evaluator actor.
        
        Args:
            config: Configuration dictionary
            gpu_id: GPU ID for this actor
        """
        self.config = config
        self.gpu_id = gpu_id
        torch.cuda.set_device(gpu_id)
        
        # Load configuration
        data_config, model_config, training_config = AutoConfig.from_config(
            config, rank=0, world_size=1, local_rank=gpu_id
        )
        
        # Create model
        self.model = AutoModel.from_config(config=model_config, local_rank=gpu_id)
        self.model.eval()
        
        # Load test dataset
        test_dataset = load_datasets(
            data_config=data_config,
            task=TaskType.EVALUATE,
            mode=ModeType.TEST
        )
        
        # Create trainer for evaluation
        self.trainer = AutoTrainer.from_config(
            model=self.model,
            train_dataset=None,
            eval_dataset=test_dataset,
            config=training_config,
            local_rank=gpu_id,
            global_rank=0,
            world_size=1,
            use_gpu=1,
        )
    
    def evaluate_user(self, user_ids: List[str]) -> Dict:
        """
        Evaluate model on a subset of users.
        
        Args:
            user_ids: List of user IDs to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        # Note: This is a simplified version. In practice, you would
        # filter the dataset by user_ids and evaluate
        results = self.trainer.evaluate()
        return results
    
    def get_stats(self) -> Dict:
        """
        Get evaluation statistics.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        results = self.trainer.evaluate()
        return results


def parse_args():
    """
    Parse command line arguments for evaluation.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate time series forecasting models using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python quito/scripts/evaluate.py \\
        --config_path configs/evaluate/patchtst/config.yaml \\
        --num_gpus 2
        """
    )

    parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to YAML config file (required)"
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for evaluation (must be >= 1)"
    )
    return parser.parse_args()


def main():
    """
    Main evaluation function.
    
    Orchestrates the evaluation process:
    1. Parse command line arguments
    2. Set up evaluation environment
    3. Load configuration
    4. Initialize Ray cluster
    5. Create evaluator actors
    6. Run distributed evaluation
    7. Aggregate and report results
    
    Raises:
        RuntimeError: If evaluation fails.
        AssertionError: If num_gpus < 1.
    """
    args = parse_args()
    
    assert args.num_gpus >= 1, "Evaluation only supports GPU, set num_gpus >= 1"
    
    # Set up evaluation environment
    config, output_dir = setup_evaluation(
        args.config_path, 
        mode=TaskType.EVALUATE
    )
    
    # Set random seed
    set_seed(config.training.seed)
    
    # Save configs
    data_config, model_config, training_config = AutoConfig.from_config(
        config, rank=0, world_size=1, local_rank=0
    )
    data_config.save(os.path.join(output_dir, 'data_config.yaml'))
    model_config.save(os.path.join(output_dir, 'model_config.yaml'))
    training_config.save(os.path.join(output_dir, 'training_config.yaml'))
    
    # Load test dataset
    test_dataset = load_datasets(
        data_config=data_config,
        task=TaskType.EVALUATE,
        mode=ModeType.TEST
    )
    logging.info(f"Test dataset size: {len(test_dataset)} samples")
    
    # Initialize Ray
    ray.init(ignore_reinit_error=True, logging_level=logging.INFO)
    
    try:
        # Create evaluator actors
        evaluators = [
            ModelEvaluator.remote(config, gpu_id=i)
            for i in range(args.num_gpus)
        ]
        
        # For simplicity, evaluate on all data with first evaluator
        # In practice, you would split the data across evaluators
        logging.info("Starting evaluation...")
        results = ray.get(evaluators[0].get_stats.remote())
        
        logging.info("=" * 80)
        logging.info("Evaluation completed successfully!")
        logging.info('The results are: \n')
        logging.info(results)
        
        # Save results
        import json
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Results saved to {results_path}")
        
    except Exception as e:
        logging.error(f"Evaluation failed: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
