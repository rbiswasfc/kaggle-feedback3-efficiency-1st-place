"""
@created by: heyao
@created at: 2022-10-26 13:21:02
copy from: https://github.com/jiawei-ren/BalancedMSE/blob/main/tutorial/balanced_mse.ipynb
"""
import torch
import torch.nn.functional as F


def weighted_mse_loss(inputs, targets, weights=None):
    loss = (inputs - targets) ** 2
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_l1_loss(inputs, targets, weights=None):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_mse_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = (inputs - targets) ** 2
    if activate == "sigmoid":
        loss *= (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    elif activate == "tanh":
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma
    elif activate is None:
        loss *= (beta * torch.abs(inputs - targets)) ** gamma
    # print(loss)
    # print(weights.expand_as(loss))
    # print(weights)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_focal_l1_loss(inputs, targets, weights=None, activate='sigmoid', beta=.2, gamma=1):
    loss = F.l1_loss(inputs, targets, reduction='none')
    if activate == "sigmoid":
        loss *= (2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1) ** gamma
    elif activate == "tanh":
        loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma
    elif activate is None:
        loss *= (beta * torch.abs(inputs - targets)) ** gamma
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def weighted_huber_loss(inputs, targets, weights=None, beta=1.):
    l1_loss = torch.abs(inputs - targets)
    cond = l1_loss < beta
    loss = torch.where(cond, 0.5 * l1_loss ** 2 / beta, l1_loss - 0.5 * beta)
    if weights is not None:
        loss *= weights.expand_as(loss)
    loss = torch.mean(loss)
    return loss


def bmc_loss(pred, target, noise_var):
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))
    loss = loss * (2 * noise_var)
    return loss


def bmse_loss(inputs, targets, noise_sigma=8.):
    return bmc_loss(inputs, targets, noise_sigma ** 2)
