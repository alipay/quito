#!/usr/bin/env python3
"""
QUITO Command Line Interface

Provides convenient commands for training, fine-tuning, evaluation, and hyperparameter tuning
of time series forecasting models.

Usage:
    quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml
    quito-cli finetune --config_path configs/finetune/patchtst/config.yaml
    quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml
    quito-cli tune --config_path configs/tune/patchtst/config.yaml
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path


def get_script_path(command: str) -> Path:
    """Get the path to the corresponding script."""
    script_dir = Path(__file__).parent / "scripts"
    script_map = {
        "pretrain": "pretrain.py",
        "finetune": "finetune.py",
        "evaluate": "evaluate.py",
        "tune": "tune.py"
    }
    
    if command not in script_map:
        raise ValueError(f"Unknown command: {command}")
    
    script_path = script_dir / script_map[command]
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    
    return script_path


def pretrain(args):
    """Run pre-training script."""
    script_path = get_script_path("pretrain")
    
    cmd = []
    
    # Check if we need distributed training
    num_gpus = getattr(args, 'num_gpus', None)
    if num_gpus and num_gpus > 1:
        # Use torchrun for distributed training
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(script_path)
        ]
    else:
        cmd = [sys.executable, str(script_path)]
    
    # Add config path
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
    
    # Add any additional arguments
    if hasattr(args, 'extra_args') and args.extra_args:
        cmd.extend(args.extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def finetune(args):
    """Run fine-tuning script."""
    script_path = get_script_path("finetune")
    
    cmd = []
    
    # Check if we need distributed training
    num_gpus = getattr(args, 'num_gpus', None)
    if num_gpus and num_gpus > 1:
        # Use torchrun for distributed training
        cmd = [
            "torchrun",
            f"--nproc_per_node={num_gpus}",
            str(script_path)
        ]
    else:
        cmd = [sys.executable, str(script_path)]
    
    # Add config path
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
    
    # Add any additional arguments
    if hasattr(args, 'extra_args') and args.extra_args:
        cmd.extend(args.extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def evaluate(args):
    """Run evaluation script."""
    script_path = get_script_path("evaluate")
    
    cmd = [sys.executable, str(script_path)]
    
    # Add config path
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
    
    # Add num_gpus if specified
    if hasattr(args, 'num_gpus') and args.num_gpus:
        cmd.extend(["--num_gpus", str(args.num_gpus)])
    
    # Add any additional arguments
    if hasattr(args, 'extra_args') and args.extra_args:
        cmd.extend(args.extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def tune(args):
    """Run hyperparameter tuning script."""
    script_path = get_script_path("tune")
    
    cmd = [sys.executable, str(script_path)]
    
    # Add config paths
    if args.config_path:
        cmd.extend(["--config_path", args.config_path])
    
    if hasattr(args, 'tuning_config_path') and args.tuning_config_path:
        cmd.extend(["--tuning_config_path", args.tuning_config_path])
    
    # Add tuning-specific parameters
    if hasattr(args, 'num_workers') and args.num_workers:
        cmd.extend(["--num_workers", str(args.num_workers)])
    
    if hasattr(args, 'num_samples') and args.num_samples:
        cmd.extend(["--num_samples", str(args.num_samples)])
    
    if hasattr(args, 'use_gpu') and args.use_gpu:
        cmd.extend(["--use_gpu", str(args.use_gpu)])
    
    # Add any additional arguments
    if hasattr(args, 'extra_args') and args.extra_args:
        cmd.extend(args.extra_args)
    
    print(f"Running command: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="QUITO Command Line Interface for Time Series Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-training
  quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml
  quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml --num_gpus 4
  
  # Fine-tuning
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml --num_gpus 4
  
  # Evaluation
  quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml
  quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml --num_gpus 2
  
  # Hyperparameter tuning
  quito-cli tune --config_path configs/tune/patchtst/config.yaml \\
                 --tuning_config_path configs/tune/patchtst/tune_config.yaml \\
                 --num_workers 4 --num_samples 100 --use_gpu 1
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Pre-train command
    pretrain_parser = subparsers.add_parser(
        "pretrain",
        help="Pre-train a model from scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml
  quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml --num_gpus 4
        """
    )
    pretrain_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the training configuration YAML file"
    )
    pretrain_parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for distributed training (uses torchrun if > 1)"
    )
    pretrain_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the script"
    )
    
    # Fine-tune command
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune a pre-trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml --num_gpus 4
        """
    )
    finetune_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the fine-tuning configuration YAML file"
    )
    finetune_parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs for distributed training (uses torchrun if > 1)"
    )
    finetune_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the script"
    )
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml
  quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml --num_gpus 2
        """
    )
    evaluate_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the evaluation configuration YAML file"
    )
    evaluate_parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for evaluation"
    )
    evaluate_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the script"
    )
    
    # Tune command
    tune_parser = subparsers.add_parser(
        "tune",
        help="Perform hyperparameter tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quito-cli tune --config_path configs/tune/patchtst/config.yaml \\
                 --tuning_config_path configs/tune/patchtst/tune_config.yaml \\
                 --num_workers 4 --num_samples 100 --use_gpu 1
        """
    )
    tune_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the base configuration YAML file"
    )
    tune_parser.add_argument(
        "--tuning_config_path",
        type=str,
        default=None,
        help="Path to the tuning configuration YAML file"
    )
    tune_parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers for hyperparameter search"
    )
    tune_parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of hyperparameter samples to try"
    )
    tune_parser.add_argument(
        "--use_gpu",
        type=int,
        default=None,
        help="Whether to use GPU for tuning (0 or 1)"
    )
    tune_parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the script"
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    command_map = {
        "pretrain": pretrain,
        "finetune": finetune,
        "evaluate": evaluate,
        "tune": tune
    }
    
    try:
        return command_map[args.command](args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
