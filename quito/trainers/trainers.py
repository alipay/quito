import torch
from typing import Dict, Tuple

from quito.trainers.base import BaseTrainer


class NaiveTrainer(BaseTrainer):
    """
    Standard trainer for time series models.
    
    This trainer provides a standard training loop for most time series models.
    """
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Perform a single training step."""
        return self.actual_model.train_step(batch)
    
    def _evaluation_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """Perform a single evaluation step."""
        
        return self.actual_model.eval_step(batch)
    