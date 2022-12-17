import re

import numpy as np
import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer

#--------------- Tokenizer ---------------------------------------------#
NEW_TOKENS = [
    "[LF]",
]


def get_tokenizer(cfg):
    """load the tokenizer"""
    tokenizer_path = cfg.model.backbone_path
    print(f"loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # NEW TOKENS
    print("adding new tokens...")
    tokens_to_add = []
    for this_tok in NEW_TOKENS:
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
    tokenizer.add_tokens(tokens_to_add)

    print(f"tokenizer len: {len(tokenizer)}")

    test_string = "This is a test \n [LF]!"
    tokenized_string = tokenizer.tokenize(test_string)
    print(f"tokenizer test: {tokenized_string}")
    return tokenizer

#--------------- Dataset ----------------------------------------------#


class FeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, cfg):
        # assign config
        self.cfg = cfg

        # label columns
        self.target_names = cfg.model.target_names

        # load tokenizer
        self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer = get_tokenizer(self.cfg)

    def pre_process(self, df):
        df["full_text"] = df["full_text"].apply(lambda x: re.sub(re.compile(r'\n\n'), " [LF] ", x))
        return df

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples["full_text"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
            return_token_type_ids=True,
        )
        return tz

    def generate_labels(self, examples):
        labels = [[] for _ in range(len(examples['input_ids']))]
        for col in self.target_names:
            for i, val in enumerate(examples[col]):
                labels[i].append(val)
        return {"labels": labels}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def get_dataset(self, df, mode='train'):
        """main api for creating the Feedback dataset
        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        df = self.pre_process(df)

        print(f"sample text:")
        print("=="*40)
        print(df.sample().full_text.values[0])
        print("=="*40)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return task_dataset
