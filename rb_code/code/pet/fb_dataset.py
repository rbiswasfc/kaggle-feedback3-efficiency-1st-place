import os
import re
from copy import deepcopy

import numpy as np
import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer

#--------------- Tokenizer ---------------------------------------------#

LABEL_INFO_MAP_0 = {
    "cohesion": "cohesion ",
    "syntax": "syntax ",
    "vocabulary": "vocabulary ",
    "phraseology": "phraseology ",
    "grammar": "grammar ",
    "conventions": "conventions ",
}


LABEL_INFO_MAP_1 = {
    "cohesion": "cohesion (essay organization; transition; logical sequencing) ",
    "syntax": "syntax (sentence structure and formation; word order) ",
    "vocabulary": "vocabulary (word diversity; topic related terms) ",
    "phraseology": "phraseology (phrases; idioms; collocations) ",
    "grammar": "grammar ",
    "conventions": "conventions (spelling; capitalization; punctuation; contractions) ",
}

LABEL_INFO_MAP_2 = {
    "cohesion": "Performance on cohesion (essay organization; transition; logical sequencing)?",
    "syntax": "Performance on syntax (sentence structure and formation; word order)?",
    "vocabulary": "Performance on vocabulary (word diversity; topic related terms)?",
    "phraseology": "Performance on phraseology (phrases; idioms; collocations)?",
    "grammar": "Performance on grammar?",
    "conventions": "Performance on conventions (spelling; capitalization; punctuation; contractions)?",
}

LABEL_INFO_MAP_3 = {
    "cohesion": "Performance on cohesion (essay organization; transition; logical sequencing)?",
    "syntax": "Performance on syntax (sentence structure and formation; word order)?",
    "vocabulary": "Performance on vocabulary (word diversity; topic related terms)?",
    "phraseology": "Performance on phraseology (phrases; idioms; collocations)?",
    "grammar": "Performance on grammar?",
    "conventions": "Performance on conventions (spelling; capitalization; punctuation; contractions)?",
}

POS_TOK_LIST_0 = [
    "good", "fine", "Good", "5",
    "strong", "positive", "right", "accurate",
    "correct", "excellent"
]

NEG_TOK_LIST_0 = [
    "bad", "poor", "Bad", "1", "weak",
    "negative", "wrong", "awful", "grim",
    "hopeless", "dire", "faulty", "inaccurate",
    "incorrect"
]

POS_TOK_LIST_1 = [
    "good", "fine", "Good", "5",
    "strong", "positive", "right", "accurate",
    "correct", "excellent"
]

NEG_TOK_LIST_1 = [
    "bad", "poor", "Bad", "1", "weak",
    "negative", "wrong", "awful", "grim",
    "hopeless", "dire", "faulty", "inaccurate",
    "incorrect"
]

POS_TOK_LIST_2 = [
    "Excellent", "Good", "A", "5",
    "10", "100", "excellent", "good",
    "High", "high", "Strong", "strong"
]
NEG_TOK_LIST_2 = [
    "1", "C", "D", "F", "Bad", "Limited", "limited", "poor",
    "Poor", "0", "Low", "low"
]

POS_TOK_LIST_3 = ["5", "10", "100", "excellent", "good", "high", "strong", "a"]
NEG_TOK_LIST_3 = ["1", "c", "d", "f", "bad", "limited", "poor", "0", "low"]


LABEL_INFO_MAP_LIST = [LABEL_INFO_MAP_0, LABEL_INFO_MAP_1, LABEL_INFO_MAP_2, LABEL_INFO_MAP_3]
POS_TOK_LIST = [POS_TOK_LIST_0, POS_TOK_LIST_1, POS_TOK_LIST_2, POS_TOK_LIST_3]
NEG_TOK_LIST = [NEG_TOK_LIST_0, NEG_TOK_LIST_1, NEG_TOK_LIST_2, NEG_TOK_LIST_3]

#------------------ tokenizer -----------------------_#


def get_tokenizer(cfg):
    """load the tokenizer"""
    tokenizer_path = cfg.model.backbone_path
    print(f"loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"tokenizer len: {len(tokenizer)}")

    test_string = "This is a test. It's all good."
    tokenized_string = tokenizer.tokenize(test_string)
    print(f"tokenizer test: {tokenized_string}")
    return tokenizer


#-------- Prompt based -----------------#

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

        # load template
        self.label_info_map = LABEL_INFO_MAP_LIST[cfg.model.pv_id]
        self.template = self.get_template()
        print("The following template will be used for PET:")
        print(self.template)

        # verbalizer
        self.pos_tok_list = POS_TOK_LIST[cfg.model.pv_id]
        self.neg_tok_list = NEG_TOK_LIST[cfg.model.pv_id]

        self.pos_tok_id_list = self.tokenizer.convert_tokens_to_ids(self.pos_tok_list)
        self.neg_tok_id_list = self.tokenizer.convert_tokens_to_ids(self.neg_tok_list)

        print(f"positive token list: {self.pos_tok_list}, id: {self.pos_tok_id_list}")
        print(f"negative token list: {self.neg_tok_list}, id: {self.neg_tok_id_list}")

    def load_tokenizer(self):
        self.tokenizer = get_tokenizer(self.cfg)

    def get_template(self):
        # add target wise prefix
        definitions = [
            f"{self.label_info_map['cohesion']}",
            f"{self.label_info_map['syntax']}",
            f"{self.label_info_map['vocabulary']}",
            f"{self.label_info_map['phraseology']}",
            f"{self.label_info_map['grammar']}",
            f"{self.label_info_map['conventions']}",
        ]

        if self.cfg.model.pv_id in [0, 1]:
            template = "Evaluate the following essay based on " + f"{self.tokenizer.mask_token}, ".join(definitions)
        elif self.cfg.model.pv_id in [2, 3]:
            template = "Student Essay Evaluation:\n\n" + f" {self.tokenizer.mask_token}. ".join(definitions)
        else:
            raise NotImplementedError

        template += f"{self.tokenizer.mask_token}. {self.tokenizer.sep_token}\n\n"
        return template

    def pre_process(self, df):
        def _process_text(text):
            paragraphs = text.split("\n\n")
            to_return = "Essay: "
            for pid, p in enumerate(paragraphs):
                to_return += f"\n[Paragraph {pid+1}] {p}"
                # to_return += f"<p> {p} </p>"
            to_return = self.template + to_return
            return to_return

        df["full_text"] = df["full_text"].apply(_process_text)

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

    def get_mask_token_idxs(self, examples):
        mask_token_idxs = []
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        for example_input_ids in examples["input_ids"]:
            example_mask_token_idxs = [pos for pos, this_id in enumerate(example_input_ids) if this_id == mask_token_id]
            mask_token_idxs.append(example_mask_token_idxs)

        return {"mask_token_idxs": mask_token_idxs}

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
        task_dataset = task_dataset.map(self.get_mask_token_idxs, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return task_dataset
