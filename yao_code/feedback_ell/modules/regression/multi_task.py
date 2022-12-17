"""
@created by: heyao
@created at: 2022-09-25 23:28:50
"""
from omegaconf import DictConfig
import torch.nn as nn

from feedback_ell.modules.base import BaseLightningModule
from feedback_ell.nn.heads.msd import MultiSampleDropout
from feedback_ell.utils.meter import AverageMeter


class FeedbackMultiTaskModule(BaseLightningModule):
    """This is the model module for regression"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self._create_head(feature_size=self.feature_size)
        self.trained_target = [0, 1, 2, 3, 4, 5]
        self.train_external_losses = AverageMeter()
        self.val_external_losses = AverageMeter()

    def _create_head(self, feature_size):
        # model_head = self.config.model.get("head", "")
        if self.config.model.msd:
            self.head = MultiSampleDropout(feature_size, 6)
        elif self.config.train.multi_task.enable:
            self.head = nn.Linear(feature_size, 6 + len(self.config.train.multi_task.tasks))
        else:
            self.head = nn.Linear(feature_size, 6)

    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        main_task, main_target = logits[:, :6], y[:, :6]
        external_task, external_target = logits[:, 6:], y[:, 6:]
        main_loss = self.compute_loss(main_task, main_target)
        external_loss = self.compute_loss(external_task, external_target)
        loss = main_loss + self.config.train.multi_task.weight * external_loss
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_external_losses.update(external_loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/external_loss", self.train_external_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        main_task, main_target = logits[:, :6], y[:, :6]
        external_task, external_target = logits[:, 6:], y[:, 6:]
        main_loss = self.compute_loss(main_task, main_target)
        external_loss = self.compute_loss(external_task, external_target)
        loss = main_loss + self.config.train.multi_task.weight * external_loss

        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_external_losses.update(external_loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/external_loss", self.val_external_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        return loss
