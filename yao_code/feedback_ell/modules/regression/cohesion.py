"""
@created by: heyao
@created at: 2022-11-16 19:15:57
"""
import numpy as np
import torch
from omegaconf import DictConfig
import torch.nn as nn
from sklearn.metrics import mean_squared_error
from feedback_ell.modules.base import BaseLightningModule
from feedback_ell.nn.poolers import MultiPooling
from feedback_ell.utils import label_columns


class NeuralTensorLayer(nn.Module):
    def __init__(self, hidden_size, k=128):
        super(NeuralTensorLayer, self).__init__()
        self.bi_linear = nn.Bilinear(hidden_size, hidden_size, k)
        self.v_product = nn.Linear(hidden_size * 2, k)
        self.bias = nn.Parameter(torch.zeros((1, )), requires_grad=True)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.u = nn.Linear(k, 1)

    def forward(self, x1, x2):
        y1 = self.bi_linear(x1, x2)
        y2 = self.v_product(torch.cat([x1, x2], dim=1)) + self.bias
        return self.sigmoid(self.u(self.tanh(y1 + y2)))


class CohesionRegressionModule(BaseLightningModule):
    """This is the model module for regression"""
    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None
        self.skip = 64
        hidden_size = self.backbone.config.hidden_size
        self.customer_pooling = MultiPooling(pooling_name="lstm", hidden_size=self.feature_size // 2)
        self.pooling = MultiPooling(pooling_name="meanmax", hidden_size=self.feature_size // 2)
        k = 64
        self.neural_tensor_layer = NeuralTensorLayer(hidden_size, k)
        self.cohesion_head = nn.LazyLinear(1)
        self.other_head = nn.Linear(hidden_size * 2, 5)

    def compute_loss(self, logits, labels, weight=None):
        if self.has_weight:
            return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target], weights=weight)
        return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"], position_ids=x.get("position_ids", None))
        step_size = feature.shape[1]
        cohesion_features = []
        skip = self.skip if step_size > self.skip else 16
        for i in range(self.config.train.max_length // self.skip - 1):
            v_a = feature[:, (1 + i * skip) % (step_size - 1)]
            v_b = feature[:, (1 + (i + 1) * skip) % (step_size - 1)]
            cohesion_features.append(self.neural_tensor_layer(v_a, v_b))
        cohesion_features = torch.cat(cohesion_features, dim=1)
        feature = self.pooling(feature)
        cohesion_feature = torch.cat([cohesion_features, feature], dim=1)
        cohesion = self.cohesion_head(cohesion_feature)
        others = self.other_head(feature)
        out = torch.cat([cohesion, others], dim=1)
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


if __name__ == '__main__':
    layer = NeuralTensorLayer(768, 5)
    x = torch.randn((2, 768))
    print(layer(x, x).shape)
