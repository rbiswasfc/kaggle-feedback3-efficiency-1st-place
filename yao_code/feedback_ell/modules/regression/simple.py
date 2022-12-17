"""
@created by: heyao
@created at: 2022-09-05 21:08:49
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


class FeedbackRegressionModule(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None

    def compute_loss(self, logits, labels, weight=None):
        if self.has_weight:
            return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target], weights=weight)
        return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"], position_ids=x.get("position_ids", None))
        out = self.head(feature)
        return out

    def training_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)
        loss = self.compute_loss(logits, y, weight)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
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
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        for i, idx in enumerate(self.trained_target):
            labels = np.array(self.val_metrics.targets)
            predictions = np.array(self.val_metrics.predictions)
            score = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
            self.log(f"val/{label_columns[idx]}", score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


import torch.nn.functional as F


def compute_kl_loss(p, q, pad_mask=None):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')

    # pad_mask is for seq-level tasks
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss


class FeedbackRDropRegressionModule(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None
        self.alpha = self.config.train.get("alpha", 5.0)
        self.dropout = nn.Dropout(0.2)

    def compute_loss(self, logits, labels, weight=None):
        if isinstance(logits, list):
            logits1, logits2 = logits
            main_loss1 = self.criterion(logits1[:, self.trained_target], labels[:, self.trained_target])
            main_loss2 = self.criterion(logits2[:, self.trained_target], labels[:, self.trained_target])
            main_loss = (main_loss1 + main_loss2) / 2
            rdrop_loss = compute_kl_loss(logits1, logits2)
            return main_loss + self.alpha * rdrop_loss
        return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])


    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"], position_ids=x.get("position_ids", None))
        out = self.head(self.dropout(feature))
        return out

    def training_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits1 = self(batch)
        logits2 = self(batch)
        loss = self.compute_loss([logits1, logits2], y, weight)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits2.detach().cpu().numpy()[:, self.trained_target])
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
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        for i, idx in enumerate(self.trained_target):
            labels = np.array(self.val_metrics.targets)
            predictions = np.array(self.val_metrics.predictions)
            score = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
            self.log(f"val/{label_columns[idx]}", score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


class FeedbackRegressionModule2(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None
        hidden_size = self.backbone.config.hidden_size
        self.num_layers = 12 if "base" in self.config.model.path else 24
        self.low_level_pooling = MultiPooling("lstm_mean", hidden_size=hidden_size)
        self.middle_level_pooling = MultiPooling("lstm_mean", hidden_size=hidden_size)
        self.low_middle_head = nn.Bilinear(self.feature_size, self.feature_size, self.feature_size // 8)
        self.middle_high_head = nn.Bilinear(self.feature_size // 8, self.feature_size, self.feature_size // 8)
        self.head = nn.Linear(hidden_size // 8, 6)

    def compute_loss(self, logits, labels, weight=None):
        if self.has_weight:
            return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target], weights=weight)
        return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        hidden_states = self.backbone(input_ids=x["input_ids"], attention_mask=x["attention_mask"]).hidden_states
        global_feature = self.customer_pooling(hidden_states[-1], mask=x["attention_mask"])
        low_level_feature = self.low_level_pooling(hidden_states[-(1 + self.num_layers // 3 * 2)], mask=x["attention_mask"])
        middle_level_feature = self.middle_level_pooling(hidden_states[-(1 + self.num_layers // 3)], mask=x["attention_mask"])
        low_middle_feature = self.low_middle_head(low_level_feature, middle_level_feature)
        middle_high_feature = self.middle_high_head(low_middle_feature, global_feature)
        # feature = self.get_feature(x["input_ids"], x["attention_mask"], position_ids=x.get("position_ids", None))
        out = self.head(middle_high_feature)
        return out

    def training_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)
        loss = self.compute_loss(logits, y, weight)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
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
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        for i, idx in enumerate(self.trained_target):
            labels = np.array(self.val_metrics.targets)
            predictions = np.array(self.val_metrics.predictions)
            score = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
            self.log(f"val/{label_columns[idx]}", score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


class FeedbackRegressionModuleWithPLWeight(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None
        self.criterion = losses.mcrmse

    def compute_loss(self, logits, labels, weight=None):
        if self.config.train.pl.get("step_weight", False) and weight is not None:
            one_epoch_steps = self.config.dataset.n_samples_in_train // self.config.train.batch_size
            total_steps = int(one_epoch_steps * self.config.train.epochs)
            weight *= (self.global_step / total_steps)
        return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target], weights=weight)

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"], position_ids=x.get("position_ids", None))
        out = self.head(feature)
        return out

    def training_step(self, batch, batch_index):
        x, y, weights = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y, weights)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        for i, idx in enumerate(self.trained_target):
            labels = np.array(self.val_metrics.targets)
            predictions = np.array(self.val_metrics.predictions)
            score = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
            self.log(f"val/{label_columns[idx]}", score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


class FeedbackRegressionWithClassifierLossModule(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.cls_criterion = nn.CrossEntropyLoss()

    def compute_loss(self, logits, labels):
        reg_loss = self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])
        cls_loss = self.cls_criterion(logits[:, self.trained_target], labels[:, self.trained_target] * 2 - 2)
        return reg_loss + cls_loss * 1.0

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


class RegressionWithMLMAuxiliaryTask(BaseLightningModule):
    """This is the model module for regression with mlm task"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        if "deberta-v3" in config.model.path:
            self.lm_head = DebertaV2OnlyMLMHead(self.backbone.config)
        else:
            print("initial lm_head with pretrained weights if ava.")
            lm_model = AutoModelForMaskedLM.from_pretrained(config.model.path)
            self.lm_head = lm_model.lm_head
        self.lm_criterion = nn.CrossEntropyLoss(reduction="none")
        self.lm_loss_weight = self.config.train.get("lm_loss_weight", 0.1)
        self.lm_losses = AverageMeter()

    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        if "mask_input_ids" in x:
            # do mlm task
            mlm_outputs = self.backbone(input_ids=x["mask_input_ids"], attention_mask=x["attention_mask"])
            prediction_scores = self.lm_head(mlm_outputs.last_hidden_state)
            vocab_size = prediction_scores.shape[-1]
            # print(prediction_scores.shape, x["mask_labels"].shape)
            masked_lm_loss = self.lm_criterion(prediction_scores.view(-1, vocab_size), x["mask_labels"].view(-1))
            masked_lm_loss = torch.masked_select(masked_lm_loss, x["mask_labels"].view(-1) != 100).mean()
            loss += self.lm_loss_weight * masked_lm_loss
            self.lm_losses.update(masked_lm_loss.item(), n=x["mask_labels"].shape[0])
            # print(masked_lm_loss)
            # print(loss)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log("train/lm_loss", self.lm_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
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
