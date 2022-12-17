"""
@created by: heyao
@created at: 2022-11-11 19:35:44
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimilarLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        return (1 - F.cosine_similarity(y_pred, y_true)).mean()


if __name__ == '__main__':
    criterion = SimilarLoss()
    y_pred = torch.rand((4, 6))
    y_true = torch.rand((4, 6))
    print(criterion(y_pred, y_true))
