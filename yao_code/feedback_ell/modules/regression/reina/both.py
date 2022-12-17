"""
@created by: heyao
@created at: 2022-10-26 11:46:54
"""
import torch
from omegaconf import DictConfig

import torch.nn as nn

from feedback_ell.modules.base import BaseLightningModule


class ReinaRegressionModule(BaseLightningModule):
    """This is the model module for regression"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.head = nn.Linear(self.feature_size, 12)

    def compute_loss(self, logits, labels):
        return self.criterion(logits[:, :6], labels[:, :6])

    def training_step(self, batch, batch_index):
        # for reina method, the input is x, y_target, y -> inputs, the retried target, our focus target/diff
        x, y, y_target = batch
        logits = self(batch)
        y_cat = torch.cat([y, y_target], dim=1)
        loss = self.compute_loss(logits, y_cat)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy(),
                                  logits.detach().cpu().numpy()[:, :6])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y, y_target = batch
        logits = self(batch)
        y_cat = torch.cat([y, y_target], dim=1)
        loss = self.compute_loss(logits, y_cat)
        self.val_losses.update(loss.item(), n=y.shape[0])
        #  + y_target.detach().cpu().numpy()
        self.val_metrics.update(y.detach().cpu().numpy(),
                                logits.detach().cpu().numpy()[:, :6])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        return loss
