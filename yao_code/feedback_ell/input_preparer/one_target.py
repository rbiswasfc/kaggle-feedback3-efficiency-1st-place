"""
@created by: heyao
@created at: 2022-10-05 17:04:40
"""
from itertools import chain
from typing import List

import pandas as pd
import numpy as np
import transformers
from omegaconf import DictConfig


from feedback_ell.input_preparer.base import BaseInputPreparer
from feedback_ell.utils import label_columns


class OneTargetRegressionInputPreparer(BaseInputPreparer):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        super().__init__(tokenizer, config)

    def prepare_input(self, df: pd.DataFrame) -> [List, List, List]:
        max_length = self.config.train.max_length
        strip_text = self.config.dataset.get("strip_text", True)
        if strip_text:
            texts = df["full_text"].str.strip().to_list()
        else:
            texts = df["full_text"].to_list()
        labels = []
        encodings = []

        def add_to_new_encoding(main_encoding, feature_encoding):
            new_encoding = {}
            for key in main_encoding:
                if key == "input_ids":
                    new_encoding[key] = [self.tokenizer.cls_token_id] + main_encoding[key] + [
                        self.tokenizer.sep_token_id] + feature_encoding[key] + [self.tokenizer.sep_token_id]
                elif key == "attention_mask":
                    new_encoding[key] = [1] * (len(main_encoding[key]) + len(feature_encoding[key]) + 3)
            return new_encoding

        for text in texts:
            encoding = self.tokenizer(text, max_length=max_length, padding=False,
                                      truncation=True, add_special_tokens=False, return_token_type_ids=False)
            for column in label_columns:
                feature_encoding = self.tokenizer(self.config.dataset.get(column), max_length=256,
                                                  padding=False, truncation=True, add_special_tokens=False,
                                                  return_token_type_ids=False)
                encodings.append(add_to_new_encoding(encoding, feature_encoding))
        text_ids = list(chain.from_iterable([i] * 6 for i in df.text_id.values))
        if label_columns[0] not in df.columns:
            return encodings, np.array(labels), text_ids
        labels = list(chain.from_iterable(df[label_columns].values.tolist()))
        return encodings, np.array(labels), text_ids


if __name__ == '__main__':
    import pandas as pd
    from transformers import AutoTokenizer
    from omegaconf import OmegaConf

    model_path = "/media/heyao/42f00068-ba8d-48dd-8719-61eda5244b8d/pretrained-models/deberta-v3-large/"
    f = "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv"
    df = pd.read_csv(f)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    config = OmegaConf.load("../../config/deberta_v3_base_reg.yaml")
    input_preparer = OneTargetRegressionInputPreparer(tokenizer, config)
    encodings, labels, text_ids = input_preparer.prepare_input(df)
    print(len(encodings), len(labels), len(text_ids))
    for i in range(6):
        print(encodings[i])
        print(labels[i])
        print(text_ids[i])
        print(tokenizer.decode(encodings[i]["input_ids"]))
