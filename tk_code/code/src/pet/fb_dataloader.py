from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


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
        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]

        target_token_idxs = [feature["target_token_idxs"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        batch["target_token_idxs"] = target_token_idxs

        batch = {k: (torch.tensor(v, dtype=torch.float32) if k in ["labels", "aux_labels"] else torch.tensor(
            v, dtype=torch.int64)) for k, v in batch.items()}
        return batch


@dataclass
class CustomDataCollatorWithPaddingMaskAug(DataCollatorWithPadding):
    """
    data collector for seq classification
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        labels = None
        if "labels" in features[0].keys():
            labels = [feature["labels"] for feature in features]
        target_token_idxs = [feature["target_token_idxs"] for feature in features]

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        batch["target_token_idxs"] = target_token_idxs

        # mask augmentation
        input_ids = torch.tensor(deepcopy(batch["input_ids"]))
        mlm_labels = input_ids.clone()

        do_not_mask_tokens = list(set(self.tokenizer.all_special_ids))
        pass_gate = [
            [0 if token_id in do_not_mask_tokens else 1 for token_id in token_id_seq] for token_id_seq in input_ids
        ]
        pass_gate = torch.tensor(pass_gate, dtype=torch.bool)

        # self.tokenizer.mask_token
        MASKING_RATIO = 0.1
        indices_mask = torch.bernoulli(torch.full(input_ids.shape, MASKING_RATIO)).bool()
        indices_mask = torch.logical_and(indices_mask, pass_gate)
        input_ids[indices_mask] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        mlm_labels[~indices_mask] = -100  # We only compute loss on masked tokens

        batch["input_ids"] = input_ids  # replaced
        batch["mlm_labels"] = mlm_labels

        batch = {k: (torch.as_tensor(v, dtype=torch.float32) if k in ["labels", "aux_labels"] else torch.as_tensor(
            v, dtype=torch.int64)) for k, v in batch.items()}
        return batch
