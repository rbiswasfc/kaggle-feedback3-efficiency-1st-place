"""
@created by: heyao
@created at: 2022-10-14 22:18:53
"""
import pickle
from typing import List

import pandas as pd
import transformers
from omegaconf import DictConfig

from feedback_ell.input_preparer.base import BaseInputPreparer
from feedback_ell.utils.tags.align_token2word import tokenize_and_align
from feedback_ell.utils.tags.encoder import TagEncoder


class ExternalTagRegressionInputPreparer(BaseInputPreparer):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        super().__init__(tokenizer, config)
        self.tag_encoder = None

    def prepare_input(self, df: pd.DataFrame) -> [List, List, List]:
        max_length = self.config.train.max_length
        # strip_text = self.config.dataset.get("strip_text", True)
        # if strip_text:
        #     texts = df["full_text"].str.strip().to_list()
        # else:
        #     texts = df["full_text"].to_list()
        external_tag_filename = self.config.train.get("external_tag_filename", None)
        if external_tag_filename is None:
            raise RuntimeError(f"must set config.train.external_tag_filename")
        with open(external_tag_filename, "rb") as f:
            data = pickle.load(f)
        # external_tags = len(data[0][0]) - 1
        text_ids = set(df.text_id.to_list())
        data = [i for i in data if i[0][-1] in text_ids]
        labels = []
        if self.label_columns[0] in df.columns:
            if self.config.train.multi_task.enable:
                labels = df[self.label_columns + self.config.train.multi_task.get("tasks", [])].values
            else:
                labels = df[self.label_columns].values
        encodings = []
        for example in data:
            words = [i[0] for i in example]
            tags = [i[1] for i in example]
            encoding, word_ids, aligned_tags = tokenize_and_align(words, tags, self.tokenizer, max_length=max_length,
                                                                  padding=False, truncation=True,
                                                                  return_token_type_ids=False)
            encoding["input_ids"] = [self.tokenizer.cls_token_id] + encoding["input_ids"] + [self.tokenizer.sep_token_id]
            encoding["attention_mask"] = [1] + encoding["attention_mask"] + [1]
            encoding["external_tag_1"] = ["[SPECIAL]"] + aligned_tags.tolist() + ["[SPECIAL]"]
            encodings.append(encoding)
        if self.tag_encoder is None:
            self.tag_encoder = TagEncoder()
            self.tag_encoder.fit_from_list([encoding["external_tag_1"] for encoding in encodings])
        for encoding in encodings:
            encoding["external_tag_1"] = self.tag_encoder.convert_texts_to_ids(encoding["external_tag_1"]).tolist()
        for encoding in encodings:
            assert len(encoding["input_ids"]) == len(encoding["external_tag_1"])
        print("num tags:", len(self.tag_encoder.label_encoder.classes_))
        return encodings, labels, df["text_id"].to_list()
