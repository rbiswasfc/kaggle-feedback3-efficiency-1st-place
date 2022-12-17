"""
@created by: heyao
@created at: 2022-09-06 19:20:57
"""
import torch
from omegaconf import DictConfig
import torch.nn as nn

from feedback_ell.modules.base import BaseLightningModule


class LabelInjectRegressionModule(BaseLightningModule):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.label_start_id = config.tokenizer.get("label_start_id", 128001)
        self.label_end_id = config.tokenizer.get("label_end_id", 128006)
        feature_size = self.get_hidden_size(self.config.model.pooling, self.backbone.config.hidden_size)
        self.head = nn.Linear(feature_size, 1)

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        bs = x["input_ids"].shape[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"])
        feature = feature[(x["input_ids"] >= self.label_start_id) & (x["input_ids"] <= self.label_end_id)]
        out = self.head(feature).reshape(bs, -1)
        return out


class LabelInjectWithAvgRegressionModule(BaseLightningModule):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        # self.label_token_ids = config.tokenizer.get("label_token_ids", None)
        # if self.label_token_ids is None:
        #     raise RuntimeError("must config label_token_ids: List")
        self.label_positions = self.config.tokenizer.label_positions
        feature_size = self.get_hidden_size(self.config.model.pooling, self.backbone.config.hidden_size)
        self.head = nn.Linear(feature_size, 1)

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        bs = x["input_ids"].shape[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"])
        # print(feature.shape, x["input_ids"].shape)
        # features = []
        # for token_ids in self.label_token_ids:
        #     print(torch.isin(x["input_ids"], torch.tensor(token_ids, device=feature.device)))
        feature = feature[:, self.label_positions, :]
        out = self.head(feature).reshape(bs, -1)
        return out
