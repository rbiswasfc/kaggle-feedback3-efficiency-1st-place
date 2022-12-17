from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding
import numpy as np
from loguru import logger

@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        #logger.debug(features)
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch = {k: (torch.tensor(v, dtype=torch.long) if k != "labels" else torch.tensor(
            v, dtype=torch.float)) for k, v in batch.items()}

        # Truncate to max len
        mask_len = int(batch["attention_mask"].sum(axis=1).max())
        for k, v in batch.items():
            batch[k] = batch[k][:, :mask_len]

        return batch


@dataclass
class CustomDataCollatorWithPaddingMaskAug(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = 512
    return_tensors = "pt"

    def __call__(self, features):

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )


        # masked augmentation
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))  # .clone()

        do_not_mask_tokens = set(self.tokenizer.all_special_ids)
        do_not_mask_tokens = list(do_not_mask_tokens)

        pass_gate = [
            [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
        ]
        pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

        # self.tokenizer.mask_token
        # 10% of the time replace token with mask token
        indices_mask = torch.bernoulli(torch.full(input_ids.shape, 0.10)).bool()
        indices_mask = torch.logical_and(indices_mask, pass_gate)
        input_ids[indices_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        batch["input_ids"] = input_ids  # replaced
        batch = {k: (torch.tensor(v, dtype=torch.long) if k != "labels" else torch.tensor(
            v, dtype=torch.float)) for k, v in batch.items()}

        # Truncate to max len
        mask_len = int(batch["attention_mask"].sum(axis=1).max())
        for k, v in batch.items():
            batch[k] = batch[k][:, :mask_len]
        return batch

