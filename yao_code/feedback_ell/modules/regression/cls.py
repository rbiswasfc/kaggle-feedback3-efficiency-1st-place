"""
@created by: heyao
@created at: 2022-10-27 18:36:14
"""
import torch
import torch.nn as nn
from omegaconf import DictConfig

from feedback_ell.modules.base import BaseLightningModule


class ClassifierRegressionModule(BaseLightningModule):
    """This is the model module for regression"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.cls_criterion = nn.CrossEntropyLoss()
        self.cls_head = nn.Linear(self.feature_size, 30)
        self.head = nn.Linear(30, 6)
        self.label_matrix = nn.Parameter(torch.FloatTensor(
            [
                [1, 0, 0, 0, 0], [0.5, 0.5, 0, 0, 0], [0, 1, 0, 0, 0],
                [0, 0.5, 0.5, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0.5, 0.5, 0],
                [0, 0, 0, 1, 0], [0, 0, 0, 0.5, 0.5], [0, 0, 0, 0, 1]
            ]), requires_grad=False)

    def compute_loss(self, logits, labels):
        cls_logits, logits = logits

        new_labels = self.label_matrix[(labels * 2 - 2).to(device=labels.device, dtype=torch.int64)]
        cls_loss = self.cls_criterion(cls_logits.reshape(-1, 5), new_labels.reshape(-1, 5))

        reg_loss = self.criterion(logits, labels)
        return cls_loss * 0.4 ** self.current_epoch + reg_loss

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"])  # meanmax out
        cls_out = self.cls_head(feature)
        out = self.head(cls_out)
        return cls_out, out

    def training_step(self, batch, batch_index):
        x, y = batch
        cls_logits, logits = self(batch)
        loss = self.compute_loss([cls_logits, logits], y)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy(),
                                  logits.detach().cpu().numpy())
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        x, y = batch
        cls_logits, logits = self(batch)
        loss = self.compute_loss([cls_logits, logits], y)
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy(),
                                logits.detach().cpu().numpy())
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


if __name__ == '__main__':
    from omegaconf import OmegaConf

    config = OmegaConf.load("../../../config/deberta_v3_large_reg.yaml")
    print(config)
    model = ClassifierRegressionModule(config)
    input_ids = torch.LongTensor([[1, 31066, 12, 312, 123, 52321, 1231, 1231, 54, 123, 1231, 12312, 412, 123, 3, 4, 2]])
    attention_mask = torch.LongTensor([[1] * input_ids.shape[1]])
    labels = torch.FloatTensor([[1.5, 2, 2, 3, 2, 1]])
    mask_input_ids = torch.LongTensor(
        [[1, 31066, 128000, 312, 123, 128000, 1231, 1231, 54, 123, 1231, 128000, 412, 123, 3, 4, 2]])
    mask_labels = torch.LongTensor(
        [[-100, -100, 12, -100, -100, 52321, -100, -100, -100, -100, -100, 12312, -100, -100, -100, -100, -100]])
    model.training_step([{
        "input_ids": input_ids, "attention_mask": attention_mask,
        "mask_input_ids": mask_input_ids, "mask_labels": mask_labels
    }, labels], 0)
    cls_logits, logits = model([{"input_ids": input_ids, "attention_mask": attention_mask}, 1])
    # print(f"{logits = }")
    # print(f"{logits.shape = }")
    # print(f"{cls_logits = }")
    # print(f"{cls_logits.shape = }")
    # print(f"{model.compute_loss([cls_logits, logits], labels) = }")
