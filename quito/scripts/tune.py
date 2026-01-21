# !/usr/bin/env python3
"""
Training script for time series forecasting models.

This script uses YAML configuration files for all training parameters.
"""
import os
import argparse
import logging
import sys
import yaml
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

from ray import train, tune
from ray.train.torch import TorchTrainer, prepare_model

from quito.config.auto import AutoConfig
from quito.config.training import TaskType, ModeType
from quito.models.auto import AutoModel
from quito.trainers.auto import AutoTrainer
from quito.utils.distributed import setup, setup_logging
from quito.utils.common import set_seed, deep_update
from quito.datasets import load_datasets

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


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
        "--tuning_config_path",
        type=str,
        required=True,
        help="Path to tuning YAML config file (required)"
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        required=True,
        default=1  # num of workers of ray cluster
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        help="Use GPU for training"
    )
    return parser.parse_args()


def worker_fn(configs):
    """
    Worker function for Ray Train distributed training.

    This function is executed by each worker in the Ray Train cluster.
    It performs the actual training loop for a single hyperparameter trial.

    Args:
        configs (dict): Dictionary containing:
            - trial_config: Trial-specific configuration (merged base + search space)
            - args: Command line arguments

    The function:
    1. Extracts distributed training context (rank, world_size)
    2. Loads and validates configuration
    3. Sets up logging and random seed
    4. Loads training and validation datasets
    5. Creates and trains the model
    6. Reports metrics and checkpoint to Ray Train
    """
    world_size = train.get_context().get_world_size()
    local_rank = train.get_context().get_local_rank()
    global_rank = train.get_context().get_world_rank()
    config = configs['trial_config']
    args = configs['args']
    data_config, model_config, training_config = AutoConfig.from_config(config, rank=global_rank, world_size=world_size,
                                                                        local_rank=local_rank)
    output_dir = training_config.output_dir
    setup_logging(global_rank, save_dir=output_dir, filename="log.txt")
    set_seed(training_config.seed + local_rank)

    # save config
    if global_rank == 0:
        data_config.save(os.path.join(output_dir, 'data_config.yaml'))
        model_config.save(os.path.join(output_dir, 'model_config.yaml'))
        training_config.save(os.path.join(output_dir, 'training_config.yaml'))

    # Init training dataset
    train_dataset = load_datasets(
        data_config=data_config,
        task=TaskType.FINE_TUNE,
        mode=ModeType.TRAIN
    )
    valid_dataset = load_datasets(
        data_config=data_config,
        task=TaskType.FINE_TUNE,
        mode=ModeType.VALID
    )
    logging.info(f"Training dataset size: {len(train_dataset)} samples")
    logging.info(f"Validation dataset size: {len(valid_dataset)} samples")
    # Create model
    model = AutoModel.from_config(config=model_config, local_rank=local_rank)
    logging.info(f"Model created: {model.__class__.__name__}")
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    # wrap model with ray train
    model = prepare_model(model)
    # Create trainer
    trainer = AutoTrainer.from_config(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        config=training_config,
        local_rank=local_rank,
        global_rank=global_rank,
        world_size=world_size,
        use_gpu=args.use_gpu,
    )
    logging.info(f"Trainer {trainer.__class__.__name__} created ...")

    results = trainer.train()
    if global_rank == 0:
        checkpoint = train.Checkpoint.from_directory(output_dir)
    else:
        checkpoint = None

    logging.info("=" * 80)
    logging.info("Training completed successfully!")
    logging.info('The results are: \n')
    logging.info(results)

    train.report(metrics=results, checkpoint=checkpoint)


def trainer_fn(config):
    """
    Trainer function for Ray Tune hyperparameter optimization.

    Creates a Ray Train trainer for a single hyperparameter trial and executes
    the training. This function is called by Ray Tune for each trial in the
    search space.

    Args:
        config (dict): Dictionary containing:
            - base_config: Base configuration object
            - search_space: Hyperparameter search space for this trial
            - args: Command line arguments

    Returns:
        Results are reported to Ray Tune via tune.report().
    """
    args = config['args']
    base_config = deepcopy(config['base_config'])
    num_workers = args.num_processes
    output_dir = tune.get_context().get_trial_dir()
    base_config['logging']['output_dir'] = output_dir

    scaling_config = train.ScalingConfig(
        num_workers=num_workers,
        use_gpu=True if args.use_gpu else False,
    )
    trial_config = deep_update(base_config, config['search_space'])
    train_loop_config = {
        'trial_config': trial_config,
        'args': args
    }

    run_config = train.RunConfig(storage_path=output_dir)
    trainer = TorchTrainer(worker_fn, scaling_config=scaling_config, run_config=run_config,
                           train_loop_config=train_loop_config)
    results = trainer.fit()
    print(results)
    tune.report(results.metrics)


def main():
    """
    Main hyperparameter tuning function using Ray Tune.

    Orchestrates hyperparameter optimization using Ray Tune:
    1. Configuration loading (base config + tuning config)
    2. Ray cluster initialization
    3. Search space definition from tuning config
    4. Ray Tune tuner creation and execution
    5. Best result extraction and reporting

    The tuning config file should define a search space using Ray Tune
    syntax (e.g., tune.grid_search, tune.choice, tune.uniform).

    Raises:
        ValueError: If required configuration is missing or invalid.
        RuntimeError: If tuning fails due to errors.

    Example:
        >>> # From command line
        >>> # python scripts/tune.py --config_path configs/tune/patchtst/config.yaml --tuning_config_path configs/tune/patchtst/tuning_space.yaml --num_samples 10
    """
    args = parse_args()
    # get tuning setup
    base_config, output_dir = setup(args.config_path, mode=TaskType.TUNE)
    # get tuning config
    with open(args.tuning_config_path, 'r') as f:
        tuning_config = yaml.safe_load(f)
    # initialize ray cluster for distributed training
    for k in tuning_config:
        for n, v in tuning_config[k].items():
            tuning_config[k][n] = eval(v)

    search_space = tuning_config
    ray.init(ignore_reinit_error=True, logging_level=logging.INFO)
    tuner = tune.Tuner(
        trainer_fn,
        param_space={'base_config': base_config, 'search_space': search_space, 'args': args},
        tune_config=tune.TuneConfig(
            metric="best_metric",
            mode="min",
            num_samples=args.num_samples,
        ),
        run_config=tune.RunConfig(name='param_tuning', storage_path=output_dir)
    )

    try:
        results = tuner.fit()
        best_result = results.get_best_result(metric='best_metric', mode='min')
        logging.info("=" * 80)
        logging.info("Tuning completed successfully!")
        logging.info(f"Best result: {best_result}")

    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()

