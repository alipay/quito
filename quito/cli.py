#!/usr/bin/env python3
"""
QUITO Command Line Interface

Provides convenient commands for training, fine-tuning, evaluation, and hyperparameter tuning
of time series forecasting models.

Usage:
    quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml --use_gpu 1 --num_processes 8
    quito-cli finetune --config_path configs/finetune/patchtst/config.yaml --use_gpu 1 --num_processes 8
    quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml --num_processes 8
    quito-cli tune --config_path configs/tune/patchtst/config.yaml --tuning_config_path configs/tune/patchtst/tuning_config.yaml --use_gpu 1 --num_processes 8
"""

import sys
import argparse
import subprocess

from importlib import resources


def get_script_name(command: str):
    """
    Map command names to their corresponding script filenames.

    Args:
        command (str): The command name (pretrain, finetune, evaluate, or tune).

    Returns:
        str: The corresponding script filename.

    Raises:
        ValueError: If the command is not recognized.
    """
    script_map = {
        "pretrain": "pretrain.py",
        "finetune": "finetune.py",
        "evaluate": "evaluate.py",
        "tune": "tune.py"
    }

    if command not in script_map:
        raise ValueError(f"Unknown command: {command}")

    return script_map[command]


def pretrain(args):
    """
    Construct command to run pre-training script with distributed training support.

    Uses torchrun for distributed training across multiple processes/GPUs.

    Args:
        args: Namespace object containing:
            - config_path (str): Path to YAML configuration file (required)
            - num_processes (int): Number of processes for distributed training
            - use_gpu (int): Whether to use GPU (0 or 1)

    Returns:
        list: Command list ready for subprocess execution.

    Raises:
        AssertionError: If config_path is not provided.
    """
    script_name = get_script_name("pretrain")

    cmd = []

    with resources.path("quito.scripts", script_name) as script_path:
        assert args.config_path, "config path must be provided !!!!"

        # Use torchrun for distributed training, it can handle cpu
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.num_processes}",
            str(script_path),
            f"--use_gpu={args.use_gpu}",
            f"--config_path={args.config_path}"]

        return cmd


def finetune(args):
    """
    Construct command to run fine-tuning script with distributed training support.

    Fine-tunes a pre-trained model on specific downstream tasks using torchrun
    for distributed training.

    Args:
        args: Namespace object containing:
            - config_path (str): Path to YAML configuration file (required)
            - num_processes (int): Number of processes for distributed training
            - use_gpu (int): Whether to use GPU (0 or 1)

    Returns:
        list: Command list ready for subprocess execution.

    Raises:
        AssertionError: If config_path is not provided.
    """
    script_name = get_script_name("finetune")

    cmd = []

    with resources.path("quito.scripts", script_name) as script_path:
        assert args.config_path, "config path must be provided !!!!"

        # Use torchrun for distributed training, it can handle cpu
        cmd = [
            "torchrun",
            f"--nproc_per_node={args.num_processes}",
            str(script_path),
            f"--use_gpu={args.use_gpu}",
            f"--config_path={args.config_path}"]

        if args.seed is not None:
            cmd.extend([f"--seed={args.seed}"])

        return cmd


def evaluate(args):
    """
    Construct command to run evaluation script on trained models.

    Evaluates model performance on test data using Ray for distributed evaluation
    across multiple GPUs.

    Args:
        args: Namespace object containing:
            - config_path (str): Path to YAML configuration file (required)
            - num_processes (int): Number of processes to use for evaluation (must be >= 1)
            - use_gpu (int) : Whether  to use gpu

    Returns:
        list: Command list ready for subprocess execution.
    """
    script_name = get_script_name("evaluate")

    cmd = [sys.executable]
    with resources.path("quito.scripts", script_name) as script_path:
        assert args.config_path, "config path must be provided !!!!"

        cmd.extend([str(script_path)])
        cmd.extend(["--config_path", args.config_path])
        cmd.extend(["--num_processes", str(args.num_processes)])
        cmd.extend(["--use_gpu", str(args.use_gpu)])

        return cmd


def tune(args):
    """
    Construct command to run hyperparameter tuning script.

    Uses Ray Tune for efficient hyperparameter search across multiple trials.
    Explores the hyperparameter space defined in the tuning configuration file.

    Args:
        args: Namespace object containing:
            - config_path (str): Path to base YAML configuration file (required)
            - tuning_config_path (str): Path to tuning-specific YAML config (required)
            - num_processes (int): Number of parallel workers for tuning
            - use_gpu (int): Whether to use GPU (0 or 1)
            - num_samples (int): Number of hyperparameter samples to try

    Returns:
        list: Command list ready for subprocess execution.

    Raises:
        AssertionError: If config_path or tuning_config_path is not provided.

    Note:
        The tuning configuration should define the search space for hyperparameters.
    """
    script_name = get_script_name('tune')

    cmd = [sys.executable]
    with resources.path("quito.scripts", script_name) as script_path:
        assert args.config_path, "config path must be provided !!!!"
        assert args.tuning_config_path, "tuning config path must be provided !!!!"

        cmd.extend([str(script_path)])
        cmd.extend(["--config_path", args.config_path])
        cmd.extend(["--tuning_config_path", args.tuning_config_path])
        cmd.extend(["--num_processes", str(args.num_processes)])
        cmd.extend(["--use_gpu", str(args.use_gpu)])
        cmd.extend(["--num_samples", str(args.num_samples)])

        return cmd


def main():
    """
    Main CLI entry point for QuitoBench commands.

    Parses command-line arguments and dispatches to the appropriate handler
    (pretrain, finetune, evaluate, or tune). Each command constructs and
    executes the corresponding script with the provided configuration.

    Returns:
        int: Exit code from the subprocess execution (0 for success).

    Raises:
        Exception: Any exceptions from the subprocess execution are propagated.

    Examples:
        >>> # Pre-train a model
        >>> quito-cli pretrain --config_path configs/pretrain/model.yaml

        >>> # Fine-tune a model
        >>> quito-cli finetune --config_path configs/finetune/model.yaml

        >>> # Evaluate a model
        >>> quito-cli evaluate --config_path configs/evaluate/model.yaml --num_processes 2

        >>> # Tune hyperparameters
        >>> quito-cli tune --config_path configs/tune/model.yaml \\
        ...     --tuning_config_path configs/tune/tuning.yaml
    """
    parser = argparse.ArgumentParser(
        description="QUITO Command Line Interface for Time Series Forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Pre-training
  quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml
  quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml --num_processes 4

  # Fine-tuning
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml --num_processes 4

  # Evaluation
  quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml
  quito-cli evaluate --config_path configs/evaluate/patchtst/config.yaml --num_processes 2

  # Hyperparameter tuning
  quito-cli tune --config_path configs/tune/patchtst/config.yaml \\
                 --tuning_config_path configs/tune/patchtst/tune_config.yaml \\
                 --num_processes 4 --num_samples 100 --use_gpu 1
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
  quito-cli pretrain --config_path configs/pretrain/patchtst/config.yaml --num_processes 4
        """
    )
    pretrain_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the training configuration YAML file"
    )
    pretrain_parser.add_argument(
        "--num_processes",
        type=int,
        default=6,
        help="Number of processes for distributed training"
    )
    pretrain_parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        help="whether to use GPU for training (0 or 1)"
    )

    # Fine-tune command
    finetune_parser = subparsers.add_parser(
        "finetune",
        help="Fine-tune a pre-trained model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml
  quito-cli finetune --config_path configs/finetune/patchtst/config.yaml --num_processes 4
        """
    )
    finetune_parser.add_argument(
        "--config_path",
        type=str,
        required=True,
        help="Path to the fine-tuning configuration YAML file"
    )
    finetune_parser.add_argument(
        "--num_processes",
        type=int,
        default=6,
        help="Number of processes for distributed training"
    )
    finetune_parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        help="whether to use GPU for training (0 or 1)"
    )
    finetune_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="The seed, this will override the seed in the config yaml"
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
        "--num_processes",
        type=int,
        default=1,
        help="Number of processes to use for evaluation"
    )
    evaluate_parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        help="whether to use gpu"
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
                 --num_processes 4 --num_samples 100 --use_gpu 1
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
        required=True,
        help="Path to the tuning configuration YAML file"
    )
    tune_parser.add_argument(
        "--num_processes",
        type=int,
        default=1,
        help="Number of parallel workers for hyperparameter search"
    )
    tune_parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of hyperparameter samples to try"
    )
    tune_parser.add_argument(
        "--use_gpu",
        type=int,
        default=1,
        help="Whether to use GPU for tuning (0 or 1)"
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
        cmd = command_map[args.command](args)  # build cmd for subprocess.run
        print(f"Running command: {' '.join(cmd)}")
        proc = subprocess.run(cmd)

        return proc.returncode

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
