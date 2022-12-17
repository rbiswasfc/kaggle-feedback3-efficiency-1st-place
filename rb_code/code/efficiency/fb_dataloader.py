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

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        if labels is not None:
            batch["labels"] = labels

        batch = {k: (torch.tensor(v, dtype=torch.float32) if k in ["labels", "aux_labels"] else torch.tensor(
            v, dtype=torch.int64)) for k, v in batch.items()}
        return batch
