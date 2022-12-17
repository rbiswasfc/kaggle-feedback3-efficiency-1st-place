from copy import deepcopy
from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class CustomDataCollatorWithPaddingPairwise(DataCollatorWithPadding):
    """
    data collector for sentence transformer training
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

        features_sent_1 = [{"input_ids": feature["input_ids_sent_1"],
                            "attention_mask": feature["attention_mask_sent_1"]} for feature in features]
        features_sent_2 = [{"input_ids": feature["input_ids_sent_2"],
                            "attention_mask": feature["attention_mask_sent_2"]} for feature in features]

        batch_sent_1 = self.tokenizer.pad(
            features_sent_1,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch_sent_2 = self.tokenizer.pad(
            features_sent_2,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch = dict()
        batch["input_ids_sent_1"] = batch_sent_1["input_ids"]
        batch["attention_mask_sent_1"] = batch_sent_1["attention_mask"]

        batch["input_ids_sent_2"] = batch_sent_2["input_ids"]
        batch["attention_mask_sent_2"] = batch_sent_2["attention_mask"]

        if labels is not None:
            batch["labels"] = labels

        batch = {k: (torch.tensor(v, dtype=torch.float32) if k in ["labels"] else torch.tensor(
            v, dtype=torch.int64)) for k, v in batch.items()}
        return batch


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
