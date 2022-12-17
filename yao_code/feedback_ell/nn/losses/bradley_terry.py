"""
@created by: heyao
@created at: 2022-11-11 19:14:27
"""
import torch
import torch.nn as nn


class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super(BradleyTerryLoss, self).__init__()

    def forward(self, pred_mean, true_mean):
        batch_size = len(pred_mean)
        true_comparison = true_mean.view(-1, 1) - true_mean.view(1, -1)
        pred_comparison = pred_mean.view(-1, 1) - pred_mean.view(1, -1)
        return torch.log(1 + torch.tril(torch.exp(-true_comparison * pred_comparison))).sum() / (
                    batch_size * (batch_size - 1) / 2)
