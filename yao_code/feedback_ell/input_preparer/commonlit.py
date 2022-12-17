"""
@created by: heyao
@created at: 2022-10-02 21:54:40
"""
from typing import List

import pandas as pd
import transformers
from omegaconf import DictConfig

from feedback_ell.input_preparer.base import BaseInputPreparer


class CommonlitInputPreparer(BaseInputPreparer):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        super().__init__(tokenizer, config)

    def prepare_input(self, df: pd.DataFrame) -> [List, List, List]:
        max_length = self.config.train.max_length
        strip_text = self.config.dataset.get("strip_text", True)
        if strip_text:
            texts = df["excerpt"].str.strip().to_list()
        else:
            texts = df["excerpt"].to_list()
        if "target" in df.columns:
            labels = df["target"].values
        else:
            labels = []
        encodings = []
        for text in texts:
            encoding = self.tokenizer(text, max_length=max_length, padding=False, truncation=True,
                                       add_special_tokens=True, return_token_type_ids=False)
            encodings.append(encoding)
        return encodings, labels, df["id"].to_list()
