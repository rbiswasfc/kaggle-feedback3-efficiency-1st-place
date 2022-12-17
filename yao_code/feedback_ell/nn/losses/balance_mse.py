"""
@created by: heyao
@created at: 2022-10-26 11:57:02
copy from: https://github.com/jiawei-ren/BalancedMSE
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal as MVN


def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))  # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

    return loss


class BMCLoss(nn.Module):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)


class MultiTargetBMCLoss(nn.Module):
    def __init__(self, init_noise_sigma):
        super().__init__()
        self._criterion = BMCLoss(init_noise_sigma=init_noise_sigma)

    def forward(self, pred, target):
        n_target = target.shape[1]
        loss = 0
        for i in range(n_target):
            loss += self._criterion(pred[:, [i]], target[:, [i]].float())
        return loss / 6


def bmc_loss_md(pred, target, noise_var):
    """Compute the Multidimensional Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, d].
      target: A float tensor of size [batch, d].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    I = torch.eye(pred.shape[-1], device=pred.device)
    # print(pred.device, target.device, I.device, (noise_var * I).device)
    logits = MVN(pred.unsqueeze(1), noise_var * I).log_prob(target.unsqueeze(0))  # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0], device=pred.device))  # contrastive-like loss
    loss = loss * (2 * noise_var)
    # loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable

    return loss
