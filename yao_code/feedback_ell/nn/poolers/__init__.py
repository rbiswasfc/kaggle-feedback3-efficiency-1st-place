"""
@created by: heyao
@created at: 2022-09-04 13:49:15
"""
import torch
import torch.nn as _nn

from .attention import AttentionPooling, GAUPooling, AttentionHeadMeanPooling
from .base import IdentityPooling, BasePooling
from .cls import CLSPooling
from .meanmax import MeanPooling, MaxPooling, MeanMaxPooling
from .weighted_layer import WeightedLayerPooling
from .rnns import LSTMPooling, GRUPooling
from .residual import ResidualLastLayersMeanPooling


def pooling_factory(pooling_name):
    pooling_mapping = {
        "lstm": LSTMPooling,
        "gru": GRUPooling,
        "mean": MeanPooling,
        "meanmax": MeanMaxPooling,
        "max": MaxPooling,
        "attention": AttentionPooling,
        "cls": CLSPooling,
        "gau": GAUPooling,
        "weighted": WeightedLayerPooling,
        "identity": IdentityPooling,
        "residualmean": ResidualLastLayersMeanPooling,
        "attnmean": AttentionHeadMeanPooling
    }
    if pooling_name not in pooling_mapping:
        raise ValueError(f"invalid pooling {pooling_name}, choose one from: {', '.join(pooling_mapping.keys())}")
    return pooling_mapping[pooling_name]


class MultiPooling(BasePooling):
    def __init__(self, pooling_name, *, hidden_size, num_hidden_layers=12, layer_start: int = 4):
        super(MultiPooling, self).__init__()
        self.pooling_name = pooling_name
        self.pooler = _nn.ModuleList([
            pooling_factory(i)(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
                               layer_start=layer_start) for i in pooling_name.split("_")
        ])

    def forward(self, x, mask=None):
        for layer in self.pooler:
            x = layer(x, mask)
        return x
