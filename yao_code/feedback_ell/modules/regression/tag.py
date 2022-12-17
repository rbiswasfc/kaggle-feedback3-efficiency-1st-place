"""
@created by: heyao
@created at: 2022-10-14 23:20:23
"""
import torch
from omegaconf import DictConfig
import torch.nn as nn

from feedback_ell.modules.base import BaseLightningModule


class TagTaskRegressionModule(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        hidden_size = self.backbone.config.hidden_size
        self.tag_lstm = nn.LSTM(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.tag_head = nn.Linear(hidden_size, 46)
        self.tag_criterion = nn.CrossEntropyLoss(reduction="none")

    def compute_loss(self, logits, labels):
        logits, tag_logits = logits
        labels, tag_labels = labels
        loss = self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])
        tag_loss = self.criterion(tag_logits.reshape(-1, 46), tag_labels.reshape(-1, 1))
        tag_loss = torch.masked_select(tag_loss, tag_labels.reshape(-1, 1) != -100).mean()
        return loss + tag_loss * self.config.train.tag_loss_weight

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        attention_mask = x["attention_mask"]
        # print(x["input_ids"])
        model_output = self.backbone(input_ids=x["input_ids"], attention_mask=attention_mask)
        if hasattr(model_output, "hidden_states"):
            hidden_states = model_output.hidden_states
        else:
            hidden_states = [model_output.last_hidden_state]
        if "weighted" in self.config.model.pooling or "residual" in self.config.model.pooling:
            pooler_output = self.customer_pooling(hidden_states, attention_mask)
        else:
            pooler_output = self.customer_pooling(hidden_states[-1], attention_mask)
        out = self.head(pooler_output)

        tag_out, _ = self.tag_lstm(model_output.last_hidden_state)
        tag_out = self.tag_head(tag_out)
        return out, tag_out

    def training_step(self, batch, batch_index):
        x, y = batch
        logits, tag_logits = self(batch)
        loss = self.compute_loss([logits, tag_logits], [y, x["external_tag_1"]])
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits, tag_logits = self(batch)
        loss = self.compute_loss([logits, tag_logits], [y, x["external_tag_1"]])
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        return loss

