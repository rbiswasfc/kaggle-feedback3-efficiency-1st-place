"""
@created by: heyao
@created at: 2022-09-04 21:57:39
"""
import torch


def mcrmse(outputs, targets, weights=None):
    if weights is None:
        weights = torch.tensor([1.0] * outputs.shape[0], device=outputs.device)
    weights = weights.reshape(-1, 1).repeat((1, 6))
    sq_diff = torch.square(targets - outputs)
    sq_diff = sq_diff * weights
    colwise_mse = torch.mean(sq_diff, dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss


if __name__ == '__main__':
    logits = torch.randn((2, 6))
    labels = torch.randn((2, 6))
    print(mcrmse(logits, labels, weights=torch.tensor([1.0, 1.0])))
    print(mcrmse(logits, labels, weights=None))
    print(mcrmse(logits, labels, weights=torch.tensor([1.0, 0.8])))
