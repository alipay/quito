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

from quito.config.training import TrainingConfig, StrategyType, OptimizerType, SchedulerType
from quito.models.base import BaseModel
from quito.utils.distributed import rank_zero_only


class BaseTrainer(ABC):
    """
    Base trainer class for all models in QUITO.
    
    This class provides common training functionality including training loops,
    evaluation, checkpointing, and logging.
    """
    
    def __init__(
        self,
        model: BaseModel,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        config: Optional[TrainingConfig] = None,
        local_rank: int = -1,
        global_rank: int = -1,
        world_size: int = -1,
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
        self.config = config or TrainingConfig()
        
        # Device management: handle CPU (-1) and GPU (>=0) cases
        self.device = f'cuda:{local_rank}' if local_rank >= 0 else 'cpu'
        
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

        # validate
        self.validate()
    
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
        validate inputs
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
        eval_strategies = []
        save_strategies = []
        logging_strategies = []

        if self.config.eval_epochs and self.config.eval_epochs >= 1:
            eval_strategies.append(StrategyType.EPOCHS)
        
        if self.config.eval_steps and self.config.eval_steps >= 1:
            eval_strategies.append(StrategyType.STEPS)

        if self.config.save_epochs and self.config.save_epochs >= 1:
            save_strategies.append(StrategyType.EPOCHS)
        
        if self.config.save_steps and self.config.save_steps >= 1:
            save_strategies.append(StrategyType.STEPS)
        
        if self.config.logging_epochs and self.config.logging_epochs >= 1:
            logging_strategies.append(StrategyType.EPOCHS)
        
        if self.config.logging_steps and self.config.logging_steps >= 1:
            logging_strategies.append(StrategyType.STEPS)
        
        self.eval_strategies = eval_strategies
        self.save_strategies = save_strategies
        self.logging_strategies = logging_strategies

    def _setup_model(self):
        self.actual_model.metrics = self.config.eval_metrics
        self.actual_model.setup_loss_fn(self.config.loss, self.config.loss_kwargs)

    def _setup_tensorboard(self):
        if self.global_rank == 0:
            self.writer = SummaryWriter(self.config.output_dir)
        
    def _setup_dataloaders(self):
        self.train_dataloader = self.get_train_dataloader(self.train_dataset)
        self.eval_dataloader = self.get_eval_dataloader(self.eval_dataset)
    
    def get_train_dataloader(self, ds: ConcatDataset):
        """
        Get dataloader for training, automatic attach distributed sampler to trainer
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
        get dataloader for evaluation
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
        load checkpoint from file
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
            train_loss, progress_bar = self._train_epoch()  
            # evaluate and perform checkpointing
            self.on_epoch_end(train_loss, progress_bar)
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
                
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Save final model
        self.save_checkpoint(valid_loss=None, prefix='last')
        
        logging.info("Training completed!")
        
        return {
            "best_metric": self.best_metric,
            "final_epoch": self.epoch,
        }
    
    def _train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        # config progress bar only on rank 0
        if self.global_rank == 0:
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {self.epoch + 1}/{self.config.num_epochs}",
                leave=False
            )
        else:
            progress_bar = self.train_dataloader
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass and loss computation with mixed precision
            if self.use_amp:
                with autocast():
                    loss = self._training_step(batch)
            else:
                loss = self._training_step(batch)
            
            loss_val = loss.item()
            total_loss += loss_val
            num_batches += 1
            
            # gradient scaling for accumulate steps
            if self.config.gradient_accumulation_steps > 1:
                loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass with gradient scaling for mixed precision
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
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
                
            # check if save, log at each step end
            self.on_batch_end(loss_val, progress_bar)
            
        loss = total_loss / num_batches if num_batches > 0 else 0.0

        return loss, progress_bar
    
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
        """Check if early stopping should be performed."""
        if not self.config.enable_early_stopping:
            return False

        if self.config.early_stopping_patience and self.patience_counter >= self.config.early_stopping_patience:
            return True

    def _should_log_train(self, location='epoch') -> bool:
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
        """Check if evaluation should be performed."""
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
        """Check if checkpoint should be saved."""
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
        """Check if the current model is the best so far."""
        if self.config.greater_is_better:
            return metric > self.best_metric
        else:
            return metric < self.best_metric
    
    @rank_zero_only
    def _log_metrics(self, timestamp, metrics: Dict[str, Any]):
        """Log metrics to various backends."""
        # log to tensorboard
        if self.writer:
            for k, v in metrics.items():
                self.writer.add_scalar(k, v, timestamp)
    
    @torch.no_grad()
    def evaluate(self, sync_score=False) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.eval_dataloader is None:
            raise ValueError("No evaluation dataset provided")
        
        self.model.eval()
        total_loss_dict = {}
        num_batches = 0
        if self.global_rank == 0:
            progress_bar = tqdm(self.eval_dataloader, desc="Evaluating", leave=False)
        else:
            progress_bar = self.eval_dataloader
        
        for idx, batch in enumerate(progress_bar):
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
        
        return total_loss_dict
    
    def on_batch_end(self, loss_val, progress_bar=None):
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
            self._log_to_progress_bar(progress_bar, {"valid_loss_step": valid_loss})
        else:
            valid_loss = None

        # check if save at current global step
        if self._should_save(location='step'):
            self.save_checkpoint(valid_loss=valid_loss)

        # log to progress bar
        self._log_to_progress_bar(progress_bar, {"train_loss_step": loss_val})
        
        
    def on_epoch_end(self, loss_val, progress_bar=None):
        # Log training step
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
                # save the checkpoint with best metric
                self.save_checkpoint(valid_loss=self.best_metric, prefix='best')
            else:
                self.patience_counter += 1
            
            # log to progress bar and tensorboard
            self._log_metrics(self.epoch, log_eval_dict)
            self._log_to_progress_bar(progress_bar, {"valid_loss_epoch": valid_loss})
        else:
            valid_loss = None

        # check if save at current epoch
        if self._should_save(location='epoch'):
            self.save_checkpoint(valid_loss=valid_loss)
        
        self._log_to_progress_bar(progress_bar, {"train_loss_epoch": loss_val})
        
    
    def _sync_metric(self, metric: Union[float, torch.Tensor]) -> float:
        """
        sync metric for all devices
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
    
    def _log_to_progress_bar(self, progress_bar, metrics: Dict[str, float]):
        if isinstance(progress_bar, tqdm):
            progress_bar.set_postfix(metrics)
    
    @property
    def train_sampler(self):
        if hasattr(self.train_dataloader, 'sampler') and self.train_dataloader.sampler is not None:
        return self.train_dataloader.sampler
        return None
    
