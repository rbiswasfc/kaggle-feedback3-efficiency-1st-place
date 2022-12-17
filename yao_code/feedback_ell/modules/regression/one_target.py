"""
@created by: heyao
@created at: 2022-10-05 18:26:43
"""
import torch.nn as nn
from omegaconf import DictConfig

from feedback_ell.modules.base import BaseLightningModule


class FeedbackOneTargetRegressionModule(BaseLightningModule):
    """This is the model module for regression"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.head = nn.Linear(self.feature_size, 1)

    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        self.val_losses.update(loss.item(), n=y.shape[0])

        self.val_metrics.update(y.detach().cpu().numpy(),
                                logits.detach().cpu().numpy())
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        return loss
