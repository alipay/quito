"""
Base trainer classes for QUITO library.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from pathlib import Path
from glob import glob

import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from quito.config.training import TrainerConfig, StrategyType, OptimizerType, SchedulerType
from quito.models.base import BaseModel
from quito.utils.distributed import rank_zero_only


class BaseTrainer(ABC):
    """
    Base trainer class for all models in QUITO.
    
    This class provides common training functionality including training loops,
    evaluation, checkpointing, and logging.
    """
    REGISTRY: dict = {}

    def __init__(
        self,
        model: BaseModel,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainerConfig] = None,
        local_rank: int = -1,
        global_rank: int = -1,
        world_size: int = -1,
        use_gpu: bool = True,
        **kwargs
    ):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
            config: Training configuration
            **kwargs: Additional arguments
        """
        self.model = model
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.local_rank = local_rank
        self.global_rank = global_rank
        self.world_size = world_size
        self.use_gpu = use_gpu
        self.config = config or TrainerConfig()
        
        # Device management: handle CPU (-1) and GPU (>=0) cases
        if self.use_gpu and self.local_rank >= 0 and self.world_size >=0 and self.global_rank >= 0:
            self.device = f'cuda:{local_rank}'
        else:
            self.device = 'cpu'
            
        # Initialize components
        self._setup_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.total_training_steps = 0
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('inf') if not self.config.greater_is_better else float('-inf')
        self.patience_counter = 0
        
        # Mixed precision training
        self.use_amp = self.config.fp16
        self.scaler = GradScaler() if self.use_amp else None
        
        # setup dataloader
        self.train_dataloader = None
        self.eval_dataloader = None
        self._setup_dataloaders()

        # setup tensorboard
        self.writer = None
        self._setup_tensorboard()

        # Set up checkpointing
        self.load_checkpoint()
        
        # Set up eval, checkpoint saving and logging strategies
        self.eval_strategies = []
        self.save_strategies = []
        self.logging_strategies = []
        self._setup_strategies()
        
        # progress bar
        self.current_progress_bar = None
        self.current_eval_progress_bar = None
        self.progress_bar_metrics = {
            "valid_loss_step": None, 
            "valid_loss_epoch": None, 
            "train_loss_step": None, 
            "train_loss_epoch": None
            }
        # validate
        self.validate()
    
    def __init_subclass__(cls, **kwargs):
        """
        Automatically register trainer subclasses in the REGISTRY.
        
        This method is called when a subclass of BaseTrainer is defined,
        automatically adding it to the REGISTRY dictionary for lookup by name.
        """
        super().__init_subclass__(**kwargs)
        # Register the subclass by its name
        BaseTrainer.REGISTRY[cls.__name__] = cls

    @property
    def actual_model(self):
        """Get the actual model, unwrapping DDP if needed."""
        return self.model.module if hasattr(self.model, 'module') else self.model
    
    def get_optimizer(self) -> torch.optim.Optimizer:
        """
        Get optimizer for training.
        
        Args:
            training_config: Training configuration
            
        Returns:
            Optimizer instance
        """
        if self.config.optimizer == OptimizerType.ADAM:
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_kwargs
            )
        elif self.config.optimizer == OptimizerType.ADAMW:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_kwargs
            )
        elif self.config.optimizer == OptimizerType.SGD:
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                **self.config.optimizer_kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def get_scheduler(self):
        """
        Get learning rate scheduler.
        
        Args:
            optimizer: Optimizer instance
            training_config: Training configuration
            
        Returns:
            Scheduler instance
        """
        if self.config.scheduler == SchedulerType.LINEAR:
            return torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                **self.config.scheduler_kwargs
            )
        elif self.config.scheduler== SchedulerType.COSINE:
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                **self.config.scheduler_kwargs
            )
        elif self.config.scheduler == SchedulerType.CONSTANT:
            return torch.optim.lr_scheduler.ConstantLR(
                self.optimizer,
                **self.config.scheduler_kwargs
            )
        elif self.config.scheduler == SchedulerType.STEP:
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                **self.config.scheduler_kwargs
            )
        else:
            return None

    def validate(self):
        """
        Validate trainer configuration and inputs.
        
        Checks that rank/world_size configuration is consistent (either all -1
        for CPU mode or all valid for GPU/distributed mode). Also validates that
        the early stopping metric is included in evaluation metrics.
        
        Raises:
            ValueError: If rank/world_size configuration is invalid.
            AssertionError: If early stopping metric is not in eval_metrics.
        """
        # Allow CPU mode (rank=-1, world_size=-1) or GPU mode (rank>=0, world_size>0)
        if not ((self.local_rank == -1 and self.global_rank == -1 and self.world_size == -1) or
                (self.local_rank >= 0 and self.global_rank >= 0 and self.world_size > 0)):
            raise ValueError(
                f"Invalid rank/world_size configuration: "
                f"local_rank={self.local_rank}, global_rank={self.global_rank}, world_size={self.world_size}. "
                f"Must be all -1 (CPU mode) or all valid (GPU mode)."
            )
        assert self.config.es_metric in self.config.eval_metrics,\
         f"{self.config.es_metric} must presented in {self.config.eval_metrics} !!!"

    def _setup_strategies(self):
        """
        Setup evaluation, checkpoint saving, and logging strategies.
        
        Configures when to perform evaluation, save checkpoints, and log metrics
        based on configuration (epoch-based or step-based strategies).
        """
        eval_strategies = []
        save_strategies = []
        logging_strategies = []

        if self.config.eval_epochs and self.config.eval_epochs >= 1:
            eval_strategies.append(StrategyType.EPOCHS)
        
        if self.config.eval_steps and self.config.eval_steps >= 1:
            eval_strategies.append(StrategyType.STEPS)

        if self.config.enable_checkpoints and self.config.save_epochs and self.config.save_epochs >= 1:
            save_strategies.append(StrategyType.EPOCHS)
        
        if self.config.enable_checkpoints and self.config.save_steps and self.config.save_steps >= 1:
            save_strategies.append(StrategyType.STEPS)
        
        if self.config.logging_epochs and self.config.logging_epochs >= 1:
            logging_strategies.append(StrategyType.EPOCHS)
        
        if self.config.logging_steps and self.config.logging_steps >= 1:
            logging_strategies.append(StrategyType.STEPS)
        
        self.eval_strategies = eval_strategies
        self.save_strategies = save_strategies
        self.logging_strategies = logging_strategies

    def _setup_model(self):
        """
        Setup model for training.
        
        Configures model metrics, loss function, and device placement.
        """
        self.actual_model.metrics = self.config.eval_metrics
        self.actual_model.setup_loss_fn(self.config.loss, self.config.loss_kwargs)
        # setup model's correct device
        self.actual_model.device = self.device

    def _setup_tensorboard(self):
        """
        Setup TensorBoard logging.
        
        Initializes SummaryWriter only on rank 0 to avoid duplicate logging
        in distributed training.
        """
        if self.global_rank == 0:
            self.writer = SummaryWriter(self.config.output_dir)
        
    def _setup_dataloaders(self):
        """
        Setup training and evaluation dataloaders.
        
        Creates dataloaders for both training and evaluation datasets.
        """
        self.train_dataloader = self.get_train_dataloader(self.train_dataset)
        self.eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
    
    def get_train_dataloader(self, ds: ConcatDataset):
        """
        Get dataloader for training with automatic distributed sampler attachment.
        
        Creates a DataLoader for training. Automatically uses DistributedSampler
        if distributed training is initialized, otherwise uses standard shuffling.
        
        Args:
            ds (ConcatDataset): Training dataset.
        
        Returns:
            DataLoader: Configured training dataloader, or None if ds is None.
        """
        if ds:
            # Use DistributedSampler only if distributed training is initialized
            if dist.is_initialized() and self.world_size > 1:
                sampler = DistributedSampler(ds, shuffle=self.config.shuffle)
                shuffle = False
            else:
                sampler = None
                shuffle = self.config.shuffle

            return DataLoader(
                ds,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                sampler=sampler,
                drop_last=self.config.drop_last
            )

        return None
    
    def get_eval_dataloader(self, ds):
        """
        Get dataloader for evaluation with automatic distributed sampler attachment.
        
        Creates a DataLoader for evaluation. Automatically uses DistributedSampler
        if distributed training is initialized. Evaluation is never shuffled.
        
        Args:
            ds: Evaluation dataset.
        
        Returns:
            DataLoader: Configured evaluation dataloader, or None if ds is None.
        """
        if ds:
            # Use DistributedSampler only if distributed training is initialized
            if dist.is_initialized() and self.world_size > 1:
                sampler = DistributedSampler(ds, shuffle=False)
            else:
                sampler = None
            
            return DataLoader(
                ds,
                batch_size=self.config.eval_batch_size if self.config.eval_batch_size else self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory,
                sampler=sampler,
                drop_last=self.config.drop_last
            )

        return None
            
    def load_checkpoint(self):
        """
        Load checkpoint from file if checkpoint_path is configured.
        
        Loads model state, optimizer state, scheduler state, and training progress
        from checkpoint file. Checkpoint is loaded to CPU first to avoid GPU
        memory spikes, then moved to the appropriate device.
        """
        checkpoint_path = self.config.checkpoint_path
        if checkpoint_path:
            # load checkpoint into cpu first, then move to device to avoid GPU memory spikes
            checkpoint = torch.load(
                checkpoint_path, 
                map_location='cpu'
            )   
            self._load_checkpoint(checkpoint)
            
            logging.info('Checkpoint loaded from {}'.format(checkpoint_path))
        else:
            logging.info('Perform training from scratch ...')

    def _load_checkpoint(self, checkpoint):
        """
        Internal method to load checkpoint state into trainer components.
        
        Args:
            checkpoint (dict): Checkpoint dictionary containing:
                - model_state_dict: Model weights
                - optimizer_state_dict: Optimizer state
                - scheduler_state_dict: Scheduler state (optional)
                - scaler_state_dict: GradScaler state (optional)
                - epoch: Last completed epoch
                - global_step: Last completed global step
        """
        self.actual_model.load(checkpoint)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step'] + 1
    
    @rank_zero_only
    def save_checkpoint(self, valid_loss, prefix='ckpt'):
        """
        Save training checkpoint (only on rank 0).
        
        Saves model state, optimizer state, scheduler state, and training progress.
        Manages checkpoint rotation to keep only the last k checkpoints.
        
        Args:
            valid_loss (float, optional): Validation loss to include in filename.
            prefix (str, optional): Checkpoint prefix ('ckpt', 'best', or 'last').
                Defaults to 'ckpt'.
        """
        ckpt_save_dir = os.path.join(self.config.output_dir, 'checkpoints')
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)
        
        # check to see if there exists checkpoints number > self.config.save_last_k, if true,
        # remove the one with oldest create time.
        ckpt_files = sorted(glob(os.path.join(ckpt_save_dir, 'ckpt_*.ckpt')), key=os.path.getctime)
        if (len(ckpt_files) >= self.config.save_last_k) and (prefix == 'ckpt'):
            oldest_ckpt = ckpt_files[0]
            os.remove(oldest_ckpt)
        
        # check to see if there already exists a best ckpt, if there is, remove it
        if prefix == 'best':
            best_ckpt_files = sorted(glob(os.path.join(ckpt_save_dir, 'best_*.ckpt')), key=os.path.getctime)
            if len(best_ckpt_files) >= 1:
                for f in best_ckpt_files:
                    os.remove(f)

        save_path = self._save_checkpoint(ckpt_save_dir, valid_loss, prefix)
        
        logging.info(f'Saved checkpoint to {save_path} ...')
            
    def _save_checkpoint(self, ckpt_save_dir, valid_loss, prefix):
        """
        Internal method to save checkpoint to disk.
        
        Args:
            ckpt_save_dir (str): Directory to save checkpoint.
            valid_loss (float, optional): Validation loss for filename.
            prefix (str): Checkpoint prefix.
        
        Returns:
            Path: Path to saved checkpoint file.
        """
        checkpoint = {
            'model_state_dict': self.actual_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'epoch': self.epoch,
            'global_step': self.global_step,
        }
        if valid_loss:
            ckpt_name = f'{prefix}_epoch={self.epoch}_step={self.global_step}_{self.config.es_metric.name}={valid_loss:.3f}.ckpt'
        else:
            ckpt_name = f'{prefix}_epoch={self.epoch}_step={self.global_step}.ckpt'
         
        save_path = Path(ckpt_save_dir) / ckpt_name
        torch.save(checkpoint, save_path)
        
        return save_path

        
    def train(self) -> Dict[str, Any]:
        """
        Train the model.
        
        Returns:
            Dictionary containing training results
        """
        logging.info("Starting training...")
        logging.info(f"Training configuration: {self.config}")
        
        for epoch in range(self.epoch, self.config.num_epochs):
            self.epoch = epoch
            if not self.train_dataloader:
                raise ValueError('Train Dataloader not provided in TRAINING MODE !')
            # set sampler epoch (only if using DistributedSampler)
            if (hasattr(self.train_dataloader, 'sampler') and 
                self.train_dataloader.sampler is not None and
                hasattr(self.train_dataloader.sampler, 'set_epoch')):
                self.train_dataloader.sampler.set_epoch(epoch)
            # Training phase
            train_loss = self._train_epoch()  
            # evaluate and perform checkpointing
            self.on_epoch_end(train_loss)
            if self.scheduler is not None:
                # Most schedulers need to step BEFORE checking early stopping
                if hasattr(self.scheduler, 'step'):
                    self.scheduler.step()
            # Early stopping
            # here, broadcast again, make use all processes terminate at the same time
            should_stop = 1 if self._should_early_stop() else 0
            if dist.is_initialized() and self.world_size > 1:
                should_stop = torch.tensor(int(should_stop), device=self.device)
                dist.broadcast(should_stop, src=0) # broadcast from rank 0 to other device
                should_stop = should_stop.item()
            else:
                should_stop = int(should_stop)
            if should_stop:
                logging.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
                
        
        # Save final model
        if self.config.enable_checkpoints:
            self.save_checkpoint(valid_loss=None, prefix='last')
        
        logging.info("Training completed!")
        
        return {
            "best_metric": self.best_metric,
            "final_epoch": self.epoch,
        }
    
    def _train_epoch(self) -> float:
        """
        Train the model for one epoch.
        
        Performs forward pass, backward pass, gradient accumulation, and
        optimizer updates for all batches in the training dataloader.
        Supports mixed precision training if enabled.
        
        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        # zero grad at the begining to prevent checkpoint
        self.optimizer.zero_grad()
        total_loss = 0.0
        num_batches = 0
        # close the progress bar 
        if self.current_progress_bar is not None:
            self.current_progress_bar.close()

        # config progress bar only on rank 0
        if self.global_rank == 0:
            self.current_progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}",
                leave=False
            )
        else:
            self.current_progress_bar = None
        
        dataloader = self.current_progress_bar if self.current_progress_bar else self.train_dataloader
        for batch_idx, batch in enumerate(dataloader):
            # Forward pass and loss computation with mixed precision
            if self.use_amp:
                with autocast():
                    loss = self._training_step(batch)
            else:
                loss = self._training_step(batch)
            
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            
            # Backward pass with gradient scaling for mixed precision
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self._optimizer_step()
                
            # check if save, log at each step end
            self.on_batch_end(loss_val)
        
        # Handle remaining accumulated gradients
        if (batch_idx + 1) % self.config.gradient_accumulation_steps != 0:
            self._optimizer_step()

        loss = total_loss / num_batches if num_batches > 0 else 0.0

        return loss
    
    def _optimizer_step(self):
        """
        Perform optimizer step with gradient clipping and scaling.
        
        Applies gradient clipping if max_grad_norm > 0, then performs
        optimizer step with appropriate scaling for mixed precision training.
        Also increments global_step counter.
        """
        # Gradient clipping
        if self.config.max_grad_norm > 0:
            if self.use_amp:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.max_grad_norm
            )

        # Optimizer step with gradient scaling
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        self.optimizer.zero_grad()
        
        self.global_step += 1

    @abstractmethod
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Perform a single training step.
        
        Args:
            batch: Training batch
            
        Returns:
            Loss tensor
        """
        pass
    
    @abstractmethod
    def _evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Evaluation batch
            
        Returns:
            The loss dict
        """
        pass

    @rank_zero_only
    def _should_early_stop(self):
        """
        Check if early stopping should be performed.
        
        Evaluates whether training should stop early based on patience counter
        and early stopping configuration. Only evaluated on rank 0 in distributed
        training to ensure consistent behavior.
        
        Returns:
            bool: True if early stopping should be triggered, False otherwise.
        """
        if not self.config.enable_early_stopping:
            return False

        if self.config.early_stopping_patience and self.patience_counter >= self.config.early_stopping_patience:
            return True

    def _should_log_train(self, location='epoch') -> bool:
        """
        Check if training metrics should be logged at the current location.
        
        Determines whether to log training metrics based on configured
        logging strategies (epoch-based or step-based) and current progress.
        
        Args:
            location (str, optional): Location to check ('epoch' or 'step').
                Defaults to 'epoch'.
        
        Returns:
            bool: True if logging should occur, False otherwise.
        """
        if not self.logging_strategies:
            return False
        elif (StrategyType.EPOCHS in self.logging_strategies) and (location == 'epoch'):
            # this method called on epoch end, so we log on epoch end
            return self.epoch % self.config.logging_epochs == 0
        elif (StrategyType.STEPS in self.logging_strategies) and (location == 'step'):
            # this method called on step end, so we log on step end
            return self.global_step % self.config.logging_steps == 0
        
        return False
    
    def _should_eval(self, location='epoch') -> bool:
        """
        Check if evaluation should be performed at the current location.
        
        Determines whether to run evaluation based on configured evaluation
        strategies (epoch-based or step-based) and current progress.
        
        Args:
            location (str, optional): Location to check ('epoch' or 'step').
                Defaults to 'epoch'.
        
        Returns:
            bool: True if evaluation should occur, False otherwise.
        """
        if not self.eval_strategies:
            return False
        elif (StrategyType.EPOCHS in self.eval_strategies) and (location == 'epoch'):
            # this method called on epoch end, so we eval  on epoch end
            return self.epoch % self.config.eval_epochs == 0
        elif (StrategyType.STEPS in self.eval_strategies) and (location == 'step'):
            # this method called on step end, so we save  on step end
            return self.global_step % self.config.eval_steps == 0
        
        return False
    
    def _should_save(self, location='step') -> bool:
        """
        Check if checkpoint should be saved at the current location.
        
        Determines whether to save a checkpoint based on configured save
        strategies (epoch-based or step-based) and current progress.
        
        Args:
            location (str, optional): Location to check ('epoch' or 'step').
                Defaults to 'step'.
        
        Returns:
            bool: True if checkpoint should be saved, False otherwise.
        """
        if not self.save_strategies:
            return False
        elif (StrategyType.EPOCHS in self.save_strategies) and (location == 'epoch'):
            # this method called on epoch end, so we save checkpoint on epoch end
            return self.epoch % self.config.save_epochs == 0
        elif (StrategyType.STEPS in self.save_strategies) and (location == 'step'):
            # this method called on step end, so we save checkpoint on step end
            return self.global_step % self.config.save_steps == 0
        
        return False
    
    @rank_zero_only
    def _is_best_model(self, metric: float) -> bool:
        """
        Check if the current model is the best so far.
        
        Compares the current metric value against the best metric seen so far,
        considering whether higher or lower values are better based on configuration.
        Only evaluated on rank 0 in distributed training.
        
        Args:
            metric (float): Current metric value to compare.
        
        Returns:
            bool: True if current metric is better than best_metric, False otherwise.
        """
        if self.config.greater_is_better:
            return metric > self.best_metric
        else:
            return metric < self.best_metric
    
    @rank_zero_only
    def _log_metrics(self, timestamp, metrics: Dict[str, Any]):
        """
        Log metrics to various backends (e.g., TensorBoard).
        
        Writes metrics to configured logging backends. Currently supports
        TensorBoard. Only logs on rank 0 in distributed training.
        
        Args:
            timestamp (int): Timestamp for the metrics (epoch or step number).
            metrics (Dict[str, Any]): Dictionary of metric names to values.
        """
        # log to tensorboard
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, timestamp)
    
    @torch.no_grad()
    def evaluate(self, sync_score=False) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Runs the model in evaluation mode over the entire evaluation dataset,
        computing all configured metrics. Supports distributed evaluation with
        metric synchronization across devices.
        
        Args:
            sync_score (bool, optional): Whether to synchronize metrics across
                devices in distributed training. Defaults to False.
        
        Returns:
            Dict[str, float]: Dictionary mapping metric names to their average values.
        
        Raises:
            ValueError: If no evaluation dataset is provided.
        """
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataset provided")
        
        self.model.eval()
        total_loss_dict = {}
        num_batches = 0
        if self.current_eval_progress_bar is not None:
            self.current_eval_progress_bar.close()

        if self.global_rank == 0:
            self.current_eval_progress_bar = tqdm(self.eval_dataloader, desc="Evaluating", leave=False)
        else:
            self.current_eval_progress_bar = None
        
        dataloader = self.current_eval_progress_bar if self.current_eval_progress_bar else self.eval_dataloader
        for idx, batch in enumerate(dataloader):
            loss_dict, predictions = self._evaluation_step(batch)
            for k, v in loss_dict.items():
                v = v.item() if torch.is_tensor(v) else v
                if k not in total_loss_dict:
                    total_loss_dict[k] = v
                else:
                    total_loss_dict[k] += v
                    
            num_batches += 1
            if self.config.save_eval_results_top_k > 0:
                save_obj = {}
                save_obj.update(loss_dict)
                save_obj.update(batch)
                save_obj['predictions'] = predictions
                self._save_eval_results(save_obj, idx)
        
        for k, v in total_loss_dict.items():
            v = v / num_batches if num_batches > 0 else 0.0
            if self.config.sync_score or sync_score:
                # sync across devices
                v = self._sync_metric(v)

            total_loss_dict[k] = v
        
        if self.current_eval_progress_bar is not None:
            self.current_eval_progress_bar.close()

        return total_loss_dict
    
    def on_batch_end(self, loss_val):
        """
        Callback executed at the end of each training batch.
        
        Handles logging, evaluation, checkpointing, and progress bar updates
        at the batch level. Checks configured strategies to determine which
        actions to perform.
        
        Args:
            loss_val (float): Loss value for the current batch.
        """
        # Log training step
        if self._should_log_train(location='step'):
            if self.config.sync_loss:
                # we sync loss step at each logging step
                loss_val = self._sync_metric(loss_val)
            
            self._log_metrics(
                self.global_step,
                {
                "global_step": self.global_step,
                "train/train_loss_step": loss_val,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            })
        
        # check if eval at current global step
        if self._should_eval(location='step'):
            total_loss_dict = self.evaluate()
            log_eval_dict = {f'valid/{k.name}_step': v for k, v in total_loss_dict.items()}            
            valid_loss = total_loss_dict[self.config.es_metric]
            # log to progress bar and tensorboard
            self._log_metrics(self.global_step, log_eval_dict)
            self.progress_bar_metrics.update({"valid_loss_step": valid_loss})
        else:
            valid_loss = None

        # check if save at current global step
        if self._should_save(location='step'):
            self.save_checkpoint(valid_loss=valid_loss)
        
        # log to progress bar
        self.progress_bar_metrics.update({"train_loss_step": loss_val})
        self._log_to_progress_bar(self.progress_bar_metrics)
        
    def on_epoch_end(self, loss_val):
        """
        Callback executed at the end of each training epoch.
        
        Handles logging, evaluation, checkpointing, early stopping checks,
        and progress bar updates at the epoch level. Manages best model
        tracking and patience counter for early stopping.
        
        Args:
            loss_val (float): Average loss value for the completed epoch.
        """
        if self._should_log_train(location='epoch'):
            if self.config.sync_loss:
                # we sync loss step at each logging step
                loss_val = self._sync_metric(loss_val)
            
            self._log_metrics(
                self.epoch,
                {
                "train/train_loss_epoch": loss_val,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "epoch": self.epoch
            })
        
        # check if eval at current global step
        if self._should_eval(location='epoch'):
            total_loss_dict = self.evaluate()
            log_eval_dict = {f'valid/{k.name}_epoch': v for k, v in total_loss_dict.items()}
            valid_loss = total_loss_dict[self.config.es_metric]
            # Check if this is the best model, this will be evaluate at rank_0 only and broadcast to all other devices
            is_best_model = 1.0 if self._is_best_model(valid_loss) else 0.0
            if dist.is_initialized() and self.world_size > 1:
                es_info = torch.tensor([is_best_model, valid_loss], device=self.device)
                dist.broadcast(es_info, src=0) # broadcast from rank 0 to other device
            else:
                es_info = torch.tensor([is_best_model, valid_loss], device=self.device)
            # fetch best model and valid loss from rank 0
            is_best_model = int(es_info[0].item())
            valid_loss = es_info[1].item()
            if is_best_model:
                self.best_metric = valid_loss
                self.patience_counter = 0
                # save the checkpoint with best metric, if enable checkpointing
                if self.config.enable_checkpoints:
                    self.save_checkpoint(valid_loss=self.best_metric, prefix='best')
            else:
                self.patience_counter += 1
            
            # log to progress bar and tensorboard
            self._log_metrics(self.epoch, log_eval_dict)
            self.progress_bar_metrics.update({"valid_loss_epoch": valid_loss})
        else:
            valid_loss = None

        # check if save at current epoch
        if self._should_save(location='epoch'):
            self.save_checkpoint(valid_loss=valid_loss)
        
        self.progress_bar_metrics.update({"train_loss_epoch": loss_val})
        self._log_to_progress_bar(self.progress_bar_metrics)
        
    def _sync_metric(self, metric: Union[float, torch.Tensor]) -> float:
        """
        Synchronize metric across all devices in distributed training.
        
        Averages the metric value across all processes in distributed training
        using all_reduce. In single-process mode, returns the metric as-is.
        
        Args:
            metric (Union[float, torch.Tensor]): Metric value to synchronize.
        
        Returns:
            float: Synchronized (averaged) metric value.
        """
        if not torch.is_tensor(metric):
            metric = torch.tensor(metric, device=self.device)
            
        # Only sync if distributed is initialized and world size > 1
        if dist.is_initialized() and self.world_size > 1:
            dist.all_reduce(metric, op=dist.ReduceOp.SUM)
            metric = metric / self.world_size
        
        metric = metric.item()
        
        return metric
    
    @rank_zero_only
    def _save_eval_results(self, results: dict, batch_idx: int):
        """
        Save evaluation results for a batch to disk.
        
        Saves evaluation results including predictions, ground truth, and metrics
        for later analysis. Manages rotation to keep only the top-k most recent
        evaluation results. Only saves on rank 0 in distributed training.
        
        Args:
            results (dict): Dictionary containing evaluation results:
                - predictions: Model predictions
                - batch data: Input batch data
                - metrics: Computed metrics
            batch_idx (int): Index of the batch being saved.
        """
        save_dir = os.path.join(self.config.output_dir, 'evaluation')
        # check and remove the oldest save file
        files = sorted(glob(os.path.join(save_dir, 'batch_*.pkl')), key=os.path.getctime)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if len(files) >= self.config.save_eval_results_top_k:
            oldest_ckpt = files[0]
            os.remove(oldest_ckpt)

        for k, v in results.items():
            # convert to cpu 
            if isinstance(v, torch.Tensor):
                results[k] = v.cpu()

        file_path = os.path.join(save_dir, f'batch_{batch_idx}.pkl')
        torch.save(results, file_path)
    
    def _log_to_progress_bar(self, metrics: Dict[str, float]):
        """
        Update the progress bar with current metrics.
        
        Updates the tqdm progress bar's postfix with the provided metrics
        for real-time monitoring during training.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names to values
                to display in the progress bar.
        """
        if self.current_progress_bar is not None:
            self.current_progress_bar.set_postfix(metrics)
    
    @property
    def train_sampler(self):
        """
        Get the training data sampler.
        
        Returns the sampler used by the training dataloader, which may be
        a DistributedSampler in distributed training or None for standard shuffling.
        
        Returns:
            Optional[Sampler]: The training sampler, or None if not using a sampler.
        """
        if hasattr(self.train_dataloader, 'sampler') and self.train_dataloader.sampler is not None:
            return self.train_dataloader.sampler
        return None

    