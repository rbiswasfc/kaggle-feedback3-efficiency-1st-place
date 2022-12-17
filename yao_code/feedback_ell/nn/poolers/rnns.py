"""
@created by: heyao
@created at: 2022-09-04 13:56:28
"""
import torch
import torch.nn as nn

from feedback_ell.nn.poolers.base import BasePooling


class LSTMPooling(BasePooling):
    def __init__(self, *, hidden_size, double_hidden_size=False, **kwargs):
        super().__init__()
        hidden_cells = hidden_size if double_hidden_size else hidden_size // 2
        self.lstm = nn.LSTM(hidden_size, hidden_cells, bidirectional=True, batch_first=True)

    def forward(self, x, mask=None):
        feature, _ = self.lstm(x)
        return feature


class GRUPooling(BasePooling):
    def __init__(self, *, hidden_size, double_hidden_size=False, **kwargs):
        super().__init__()
        hidden_cells = hidden_size if double_hidden_size else hidden_size // 2
        self.lstm = nn.GRU(hidden_size, hidden_cells, bidirectional=True, batch_first=True)

    def forward(self, x, mask=None):
        feature, _ = self.lstm(x)
        return feature


if __name__ == '__main__':
    x = torch.rand((4, 128, 768))
    pooling = LSTMPooling(hidden_size=768)
    print(pooling(x).shape)
