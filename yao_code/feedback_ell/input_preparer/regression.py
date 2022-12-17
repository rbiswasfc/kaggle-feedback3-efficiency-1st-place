"""
@created by: heyao
@created at: 2022-09-05 19:43:44
"""
from collections import Counter
from typing import List

import pandas as pd
import transformers
from omegaconf import DictConfig

from feedback_ell.input_preparer.base import BaseInputPreparer


class RegressionInputPreparer(BaseInputPreparer):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        super().__init__(tokenizer, config)

    def prepare_input(self, df: pd.DataFrame) -> [List, List, List]:
        max_length = self.config.train.max_length
        strip_text = self.config.dataset.get("strip_text", True)
        if strip_text:
            texts = df["full_text"].str.strip().to_list()
        else:
            texts = df["full_text"].to_list()
        # new_features = pd.read_csv("/home/heyao/kaggle/feedback-ells/input/train_grammar_related_features.csv")
        # new_features["grammar_error_ratio"] = new_features["GRAMMAR"] / new_features["n_words"]
        # new_features["total_error_ratio"] = new_features["total_errors"] / new_features["n_words"]
        # new_features = new_features.fillna(0.0)
        if self.label_columns[0] in df.columns:
            if self.config.train.multi_task.enable:
                labels = df[self.label_columns + self.config.train.multi_task.get("tasks", [])].values
            else:
                labels = df[self.label_columns].values
        else:
            labels = []
        encodings = []
        for i, text in enumerate(texts):
            encoding = self.tokenizer(text, max_length=max_length, padding=False, truncation=True,
                                      add_special_tokens=True, return_token_type_ids=False)
            # encoding["stat"] = new_features.loc[i, ["grammar_error_ratio", "total_error_ratio"]].to_list()
            encodings.append(encoding)
        return encodings, labels, df["text_id"].to_list()


class TokenRegressionInputPreparer(BaseInputPreparer):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        super().__init__(tokenizer, config)

    def prepare_input(self, df: pd.DataFrame) -> [List, List, List]:
        max_length = self.config.train.max_length
        strip_text = self.config.dataset.get("strip_text", True)
        if strip_text:
            texts = df["full_text"].str.strip().to_list()
        else:
            texts = df["full_text"].to_list()
        if self.label_columns[0] in df.columns:
            if self.config.train.multi_task.enable:
                labels = df[self.label_columns + self.config.train.multi_task.get("tasks", [])].values
            else:
                labels = df[self.label_columns].values
        else:
            labels = []
        encodings = []
        new_labels = []
        for i, text in enumerate(texts):
            encoding = self.tokenizer(text, max_length=max_length, padding=False, truncation=True,
                                      add_special_tokens=True, return_token_type_ids=False)
            encodings.append(encoding)
            if labels != []:
                n_tokens = len(encoding["input_ids"]) - 2
                labels[i] /= n_tokens
                new_label = labels[i].reshape(1, -1).repeat(n_tokens, axis=0)
                new_label[[0, -1], :] = -100
                new_labels.append(new_label)
        return encodings, new_labels, df["text_id"].to_list()


def get_position_ids(paragraph_ids):
    counter = Counter(paragraph_ids)
    position_ids = []
    for i in range(len(counter)):
        if counter[i] // 510 == 0:
            position_ids += list(range(1, counter[i] + 1))
        else:
            position_ids += list(range(1, 510 + 1))
            position_ids += list(range(1, counter[i] - 510))
    return position_ids


class NoParagraphOrderRegressionInputPreparer(BaseInputPreparer):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        super().__init__(tokenizer, config)

    def prepare_input(self, df: pd.DataFrame) -> [List, List, List]:
        max_length = self.config.train.max_length
        strip_text = self.config.dataset.get("strip_text", True)
        if strip_text:
            texts = df["full_text"].str.strip().to_list()
        else:
            texts = df["full_text"].to_list()
        if self.label_columns[0] in df.columns:
            if self.config.train.multi_task.enable:
                labels = df[self.label_columns + self.config.train.multi_task.get("tasks", [])].values
            else:
                labels = df[self.label_columns].values
        else:
            labels = []
        encodings = []
        for i, text in enumerate(texts):
            text = text.split("\n\n")
            # print(text)
            encoding = self.tokenizer(text, max_length=max_length, padding=False, truncation=True,
                                      add_special_tokens=False, return_token_type_ids=False, is_split_into_words=True)
            # print("num_paragraph:", len(text), "word_ids:", encoding.word_ids())
            # encoding["stat"] = new_features.loc[i, ["grammar_error_ratio", "total_error_ratio"]].to_list()
            paragraph_ids = encoding.word_ids()
            position_ids = get_position_ids(paragraph_ids)
            # position_ids = [0] * len(encoding["input_ids"])
            # encoding["position_ids"] = [1] + position_ids + [position_ids[-1] + 1]
            encoding["input_ids"] = [self.tokenizer.cls_token_id] + encoding["input_ids"] + [self.tokenizer.sep_token_id]
            encoding["attention_mask"] = [1] + encoding["attention_mask"] + [1]
            encodings.append(encoding)
        return encodings, labels, df["text_id"].to_list()
