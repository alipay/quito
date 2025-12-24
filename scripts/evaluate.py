# !/usr/bin/env python3
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
from torch.utils.data import DataLoader
import ray
from typing import Dict, Any
import traceback
from copy import deepcopy
import time
import json

from quito.config.auto import AutoConfig
from quito.config.training import TaskType, ModeType
from quito.models.auto import AutoModel
from quito.utils.distributed import setup_evaluation
from quito.utils.common import set_seed
from quito.datasets import load_datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test time series forecasting models using YAML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
    Examples:
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
        required=False,
        default=None  # num of gpus of ray cluster
    )

    return parser.parse_args()


@ray.remote
class ModelEvaluator:
    """
    Ray Actor for distributed model evaluation.

    Keeps model in GPU memory across multiple evaluations.
    """

    def __init__(self, model_config: DictConfig, training_config: DictConfig):
        """Initialize actor with model loaded once."""
        # Set seed
        set_seed(training_config.seed)
        # Load model once (stays in GPU memory)
        self.model = AutoModel.from_config(model_config, local_rank=-1)
        # load from checkpoint
        # setup evaluation metrics
        # Setup device
        if torch.cuda.is_available():
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'

        self.model.device = self.device
        self.model = self.model.to(self.device)
        if training_config.checkpoint_path:
            self.model.load(training_config.checkpoint_path)
            print(f'Model loaded from checkpoint {training_config.checkpoint_path}')

        self.model.metrics = training_config.eval_metrics

        # Store config
        self.training_config = training_config
        self.batch_size = training_config.batch_size

        # Tracking statistics
        self.eval_count = 0
        self.total_time = 0

    def evaluate_user(self, user_id: int, dataset: Any) -> Dict[str, float]:
        """
        Evaluate a single user's dataset.

        Args:
            user_id: User identifier

        Returns:
            Dictionary of metrics
        """
        try:
            start_time = time.time()
            # get data
            dataset = deepcopy(dataset)
            # modify dataset inplace to create user-specific dataset
            dataset.select_user_data(user_id)
            # Create dataloader
            dl = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False
            )

            # Initialize metrics
            total_loss_dict = {}
            n_samples = 0

            # Evaluate batches
            with torch.no_grad():
                for batch_idx, batch in enumerate(dl):
                    # Evaluate step
                    loss_dict, predictions = self.model.eval_step(batch)
                    batch_size = len(predictions)

                    # Accumulate losses
                    for k, v in loss_dict.items():
                        v = v.item() if torch.is_tensor(v) else v
                        v_scaled = v * batch_size

                        if k.name not in total_loss_dict:
                            total_loss_dict[k.name] = v_scaled
                        else:
                            total_loss_dict[k.name] += v_scaled

                    n_samples += batch_size

            # Average metrics
            avg_metrics = {}
            for k, v in total_loss_dict.items():
                avg_metrics[k] = v / n_samples if n_samples > 0 else 0.0

            # Update tracking
            elapsed = time.time() - start_time
            self.eval_count += 1
            self.total_time += elapsed

            print(
                f"Evaluated user {user_id}: {avg_metrics} "
                f"(took {elapsed:.2f}s)"
            )

            return {
                "user_id": user_id,
                "metrics": avg_metrics,
                "n_samples": n_samples,
                "eval_time": elapsed
            }

        except Exception as e:
            logging.error(f"Error evaluating user {user_id}: {str(e)}")
            logging.error(f"[User id {user_id}] Traceback:\n{traceback.format_exc()}")

            return {
                "user_id": user_id,
                "error": str(e),
                "metrics": {},
                "n_samples": 0
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get actor statistics."""
        return {
            "eval_count": self.eval_count,
            "total_time": self.total_time,
            "avg_time_per_eval": self.total_time / self.eval_count if self.eval_count > 0 else 0
        }


def main():
    """Main testing function."""
    args = parse_args()
    # get evaluation setup
    config, output_dir = setup_evaluation(args.config_path)
    # load config
    data_config, model_config, training_config = AutoConfig.from_config(config=config, rank=-1, world_size=-1,
                                                                        local_rank=-1)
    # initialize ray cluster for distributed training
    ray.init(num_gpus=args.num_gpus)
    try:
        datasets = load_datasets(
            data_config=data_config,
            task=TaskType.EVALUATE,
            mode=ModeType.TEST,
            cleanup=False,
            concat=False,
        )
        dataset_refs = [ray.put(ds) for ds in datasets]
        if args.num_gpus > 0:
            evaluator_cls = ModelEvaluator.options(num_gpus=1)
        else:
            evaluator_cls = ModelEvaluator
        # Create evaluator actors (one per GPU)
        num_workers = max(1, args.num_gpus)
        logging.info(f"Creating {num_workers} evaluator actors")
        evaluators = [
            evaluator_cls.remote(model_config, training_config)
            for _ in range(num_workers)
        ]

        # Prepare evaluation tasks
        tasks = []
        for dataset_idx, dataset in enumerate(datasets):
            user_ids = dataset.get_all_ids()
            for user_id in user_ids:
                tasks.append((dataset_idx, int(user_id)))

        logging.info(f"Total tasks to evaluate: {len(tasks)}")

        # Distribute tasks round-robin across evaluators
        futures = []
        results_metadata = {}

        for task_idx, (dataset_idx, user_id) in enumerate(tasks):
            evaluator_id = task_idx % num_workers
            evaluator = evaluators[evaluator_id]
            future = evaluator.evaluate_user.remote(user_id, dataset_refs[dataset_idx])
            futures.append(future)
            results_metadata[len(futures) - 1] = {
                "dataset_idx": dataset_idx,
                "user_id": user_id,
                "evaluator_id": evaluator_id
            }

        # Collect results with progress tracking
        logging.info("Starting evaluation")
        results = []

        for future_idx, future in enumerate(futures):
            try:
                result = ray.get(future, timeout=300)  # 5 min timeout per task
                results.append(result)

                if (future_idx + 1) % max(1, len(futures) // 10) == 0:
                    logging.info(f"Progress: {future_idx + 1}/{len(futures)} evaluations complete")

            except ray.exceptions.GetTimeoutError:
                logging.error(f"Task {future_idx} timed out")
                results.append({
                    "user_id": results_metadata[future_idx]["user_id"],
                    "error": "Timeout",
                    "metrics": {}
                })
            except Exception as e:
                logging.error(f"Task {future_idx} failed: {str(e)}")
                results.append({
                    "user_id": results_metadata[future_idx]["user_id"],
                    "error": str(e),
                    "metrics": {}
                })

        with open(os.path.join(output_dir, 'eval_results.json'), 'w') as f:
            json.dump({'final_results': results}, f)

    finally:
        ray.shutdown()
        logging.info("Ray cluster shut down")


if __name__ == "__main__":
    main()

