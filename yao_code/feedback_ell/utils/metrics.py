"""
@created by: heyao
@created at: 2022-09-05 20:55:34
"""
import numpy as np


def competition_score(labels, logits):
    colwise_mse = np.mean(np.square(labels - logits), axis=0)
    loss = np.mean(np.sqrt(colwise_mse), axis=0)
    return loss
