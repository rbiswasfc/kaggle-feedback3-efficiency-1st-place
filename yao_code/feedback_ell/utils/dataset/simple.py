"""
@created by: heyao
@created at: 2022-09-02 20:15:07
"""
import numpy as np
import torch

from feedback_ell.utils.reweight import _prepare_weights, preparer_weights_for_multi_label
from feedback_ell.utils.aug.mask import regular_mask_aug_without_pad, mask_tokens


class CompetitionDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, weights=None, mask_ratio=0, tokenizer=None, reweight=None):
        self.x = x
        self.y = y
        self.weights = weights
        self.mask_ratio = mask_ratio
        self.tokenizer = tokenizer
        if mask_ratio != 0 and tokenizer is None:
            raise RuntimeError("mask aug must input tokenizer")
        self.reweight = reweight
        if reweight is not None and y is not None:
            self.re_weights = preparer_weights_for_multi_label(y, reweight=reweight, max_target=9)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        if self.mask_ratio:
            x["input_ids"] = regular_mask_aug_without_pad(np.array(x["input_ids"]),
                                                          self.tokenizer, mask_ratio=self.mask_ratio).tolist()
        if self.y is None:
            return x
        if self.reweight is None and self.weights is None:
            return x, self.y[item]
        if self.weights is not None:
            weight = self.weights[item]
        else:
            weight = self.re_weights[item]
        # print(x, self.y[item], weight)
        return x, self.y[item], weight


class ReinaDataset(torch.utils.data.Dataset):
    def __init__(self, x, y_target, y=None, mask_ratio=0, tokenizer=None):
        self.x = x
        self.y_target = y_target
        self.y = y
        self.mask_ratio = mask_ratio
        self.tokenizer = tokenizer
        if mask_ratio != 0 and tokenizer is None:
            raise RuntimeError("mask aug must input tokenizer")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x = self.x[item]
        y_target = self.y_target[item]
        if self.mask_ratio:
            x["input_ids"] = regular_mask_aug_without_pad(np.array(x["input_ids"]),
                                                          self.tokenizer, mask_ratio=self.mask_ratio).tolist()
        if self.y is None:
            return x, y_target
        return x, y_target, self.y[item]


class AddMaskTaskDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, x, y, mask_ratio=0.15):
        self.x = x
        self.y = y
        self.mlm_probability = mask_ratio
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.x)

    def __getitem__(self, item):
        x, y = self.x[item], self.y[item]
        mask_x, mask_y = mask_tokens(self.tokenizer, torch.LongTensor([x["input_ids"]]), mlm_probability=self.mlm_probability)
        x["mask_input_ids"] = mask_x.squeeze(0).detach().numpy().tolist()
        x["mask_labels"] = mask_y.squeeze(0).detach().numpy().tolist()
        return x, y
