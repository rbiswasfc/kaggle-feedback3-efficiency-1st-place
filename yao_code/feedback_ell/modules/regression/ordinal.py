"""
@created by: heyao
@created at: 2022-11-25 21:29:19
"""
import numpy as np
import torch
from omegaconf import DictConfig
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from transformers import AutoModelForMaskedLM
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2OnlyMLMHead

from feedback_ell.modules.base import BaseLightningModule
from feedback_ell.nn import losses
from feedback_ell.nn.heads.mlm import DebertaOnlyMLMHead
from feedback_ell.nn.poolers import MultiPooling
from feedback_ell.utils import label_columns
from feedback_ell.utils.meter import AverageMeter


class OrdinalRegressionModule(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None
        self.label_matrix = nn.Parameter(torch.FloatTensor(
            [
                [1, 0, 0, 0, 0], [1.0, 0.5, 0, 0, 0], [1, 1, 0, 0, 0],
                [1, 1, 0.5, 0, 0], [1, 1, 1, 0, 0], [1, 1, 1, 0.5, 0],
                [1, 1, 1, 1, 0], [1, 1, 1, 1, 0.5], [1, 1, 1, 1, 1]
            ]), requires_grad=False)
        self.criterion = nn.BCEWithLogitsLoss()
        self.head1 = nn.Linear(self.feature_size, 5)
        self.head2 = nn.Linear(self.feature_size, 5)
        self.head3 = nn.Linear(self.feature_size, 5)
        self.head4 = nn.Linear(self.feature_size, 5)
        self.head5 = nn.Linear(self.feature_size, 5)
        self.head6 = nn.Linear(self.feature_size, 5)

    def gen_label(self, labels):
        with torch.no_grad():
            new_labels = self.label_matrix[(labels * 2 - 2).to(device=labels.device, dtype=torch.int64)]
        return new_labels

    def compute_loss(self, logits, labels, weight=None):
        # if self.has_weight:
        #     return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target], weights=weight)
        new_labels = self.gen_label(labels)  # [B, 6, 5]
        B = new_labels.shape[0]
        loss = self.criterion(logits[:, self.trained_target, :].reshape(B, -1),
                              new_labels[:, self.trained_target].reshape(B, -1))
        return loss

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"], position_ids=x.get("position_ids", None))
        out = torch.cat([
            self.head1(feature).unsqueeze(1),
            self.head2(feature).unsqueeze(1),
            self.head3(feature).unsqueeze(1),
            self.head4(feature).unsqueeze(1),
            self.head5(feature).unsqueeze(1),
            self.head6(feature).unsqueeze(1)
        ], dim=1)
        return out

    def training_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)  # [B, 6, 5]
        loss = self.compute_loss(logits, y, weight)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.sigmoid().sum(axis=-1).detach().cpu().numpy()[:, self.trained_target])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)
        loss = self.compute_loss(logits, y, weight=weight)
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                logits.sigmoid().sum(axis=-1).detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        for i, idx in enumerate(self.trained_target):
            labels = np.array(self.val_metrics.targets)
            predictions = np.array(self.val_metrics.predictions)
            score = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
            self.log(f"val/{label_columns[idx]}", score, prog_bar=True, on_step=False, on_epoch=True)
        return loss
