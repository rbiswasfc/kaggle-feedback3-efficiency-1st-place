"""
@created by: heyao
@created at: 2022-11-11 19:40:00
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from feedback_ell.nn.losses.sim import SimilarLoss


class MultiLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.cosine = SimilarLoss()

    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true) * 0.5 + self.cosine(y_pred, y_true) * 0.5


if __name__ == '__main__':
    criterion = SimilarLoss()
    y_pred = torch.rand((4, 6))
    y_true = torch.rand((4, 6))
    print(criterion(y_pred, y_true))
