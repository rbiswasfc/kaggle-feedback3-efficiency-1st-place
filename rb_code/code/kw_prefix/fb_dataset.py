import json
import re

import numpy as np
import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer


#--------------- Sequence Ids ------------------------------------------#
def get_sequence_ids(input_ids, tokenizer):
    """
    This function derives sequence ids for a given tokenizer based on token input ids
    :param input_ids: token input id sequence
    :type input_ids: List[int]
    :param tokenizer: HF tokenizer
    :type tokenizer: PreTrainedTokenizer
    :return: sequence ids
    :rtype: List
    """
    sequence_ids = [0]*len(input_ids)

    switch = False
    special_token_ids = set(
        tokenizer.convert_tokens_to_ids(
            tokenizer.special_tokens_map.values()
        )
    )
    for i, input_id in enumerate(input_ids):
        if input_id == tokenizer.sep_token_id:
            switch = True
        if switch:
            sequence_ids[i] = 1
        if input_id in special_token_ids:
            sequence_ids[i] = -1
    return sequence_ids


#--------------- Tokenizer ---------------------------------------------#
NEW_TOKENS = [
    "[LF]",
    "[SOE]",
    "[EOE]",
]


def get_tokenizer(cfg):
    """load the tokenizer"""
    tokenizer_path = cfg.model.backbone_path
    print(f"loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        print("adding pad token")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # NEW TOKENS
    if cfg.model.add_new_tokens:
        print("adding new tokens...")
        tokens_to_add = []
        for this_tok in NEW_TOKENS:
            tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
        tokenizer.add_tokens(tokens_to_add)

    print(f"tokenizer len: {len(tokenizer)}")

    test_string = "[SOE] This is a test \n [LF] [EOE]!!"
    tokenized_string = tokenizer.tokenize(test_string)
    print(f"tokenizer test: {tokenized_string}")
    return tokenizer

#--------------- Prefix --------------------------------------------------#


def get_kw_bank(cfg):
    """
    get the keywords dataset
    """
    with open(cfg.kw_bank.cohesion, "r") as f:
        kw_cohesion = json.load(f)

    with open(cfg.kw_bank.syntax, "r") as f:
        kw_syntax = json.load(f)

    with open(cfg.kw_bank.vocabulary, "r") as f:
        kw_vocabulary = json.load(f)

    with open(cfg.kw_bank.phraseology, "r") as f:
        kw_phraseology = json.load(f)

    with open(cfg.kw_bank.grammar, "r") as f:
        kw_grammar = json.load(f)

    with open(cfg.kw_bank.conventions, "r") as f:
        kw_conventions = json.load(f)

    kw_list = [
        kw_cohesion,
        kw_syntax,
        kw_vocabulary,
        kw_phraseology,
        kw_grammar,
        kw_conventions,
    ]

    fb3_kws = set()
    for ks in kw_list:
        fb3_kws.update(ks)

    fb3_kws = list(fb3_kws)

    filtered_fb3_kws = []
    for kw in fb3_kws:
        if (len(kw) >= 3):
            filtered_fb3_kws.append(kw)

    contractions = [k for k in filtered_fb3_kws if "'" in k]
    updated_contractions = ["\'".join([x.strip() for x in t.split("'")]) for t in contractions]
    filtered_fb3_kws.extend(updated_contractions)
    return filtered_fb3_kws


def compute_prefix(text, bank):
    """
    computes prefix for pre-processing based on keyword dataset bank
    """
    text = text.lower()
    text_words = set(text.split(" "))

    unordered_prefix = ""

    for candidate in bank:
        if candidate in text:
            if candidate in unordered_prefix:
                continue
            elif len(candidate.split(" ")) == 1:  # one word
                if candidate not in text_words:
                    continue
            else:
                unordered_prefix += f"[=SEP=] {candidate}"

    segments = unordered_prefix.split("[=SEP=]")
    segments = [s.strip() for s in segments]

    ordering = [(text.index(s), s) for s in segments]
    ordering = sorted(ordering, key=lambda x: x[0])
    ordered_segments = [o[1] for o in ordering]
    to_return = ", ".join(ordered_segments)
    to_return = to_return[1:].strip()  # remove leading comma

    if len(to_return) <= 2:
        print("warning! no prefix is detected...")
        to_return = "no keywords"

    return to_return


def compute_prefix_beta(text, bank):
    """
    computes prefix for pre-processing based on keyword dataset bank
    """
    text = text.lower()
    text_words = set(text.split(" "))

    unordered_prefix = ""

    for candidate in bank:
        if len(candidate.split(" ")) == 1:  # one word
            if len(candidate) <= 8:
                continue

            candidate = f" {candidate} "

            if candidate in text:
                if candidate in unordered_prefix:
                    continue
                else:
                    unordered_prefix += f"[=SEP=] {candidate}"
        else:  # phrase units
            candidate = f"{candidate} "

            if candidate in text:
                if candidate in unordered_prefix:
                    continue
                else:
                    unordered_prefix += f"[=SEP=] {candidate}"

    segments = unordered_prefix.split("[=SEP=]")
    segments = [s.strip() for s in segments]

    ordering = [(text.index(s), s) for s in segments]
    ordering = sorted(ordering, key=lambda x: x[0])
    ordered_segments = [o[1] for o in ordering]
    to_return = ", ".join(ordered_segments)
    to_return = to_return[1:].strip()  # remove leading comma

    if len(to_return) <= 2:
        print("warning! no prefix is detected...")
        to_return = "no keywords"

    return to_return

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

        # get keyword bank
        bank = get_kw_bank(cfg)
        self.bank = sorted(bank, key=lambda x: len(x), reverse=True)

    def load_tokenizer(self):
        self.tokenizer = get_tokenizer(self.cfg)

        # additional tokens
        self.new_token_ids = set()
        if self.cfg.model.add_new_tokens:
            self.new_token_ids = set(self.tokenizer.convert_tokens_to_ids(NEW_TOKENS))

    def pre_process(self, df):
        # compute prefix
        print("computing text prefix...")
        df["prefix"] = df["full_text"].apply(lambda x: compute_prefix(x, self.bank))
        # df["prefix"] = df["full_text"].apply(lambda x: compute_prefix_beta(x, self.bank))

        # process full text
        if self.cfg.model.add_new_tokens:
            df["full_text"] = df["full_text"].apply(lambda x: re.sub(re.compile(r'\n\n'), " [LF] ", x))
            df["full_text"] = df["full_text"].apply(lambda x: " ".join(["[SOE]", x, "[EOE]"]))

        # add in prefix
        df["full_text"] = df[["prefix", "full_text"]].apply(lambda x: f"{x[0]} {self.tokenizer.sep_token} {x[1]}", axis=1)

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

    def generate_aux_labels(self, examples):
        aux_labels = []
        for ex_labels in examples["labels"]:
            ex_aux_labels = [int(2.0*l-2.0) for l in ex_labels]
            aux_labels.append(ex_aux_labels)
        return {"aux_labels": aux_labels}

    def add_sequence_ids(self, examples):
        sequence_ids = []
        input_ids = examples["input_ids"]

        for tok_ids in input_ids:
            sequence_ids.append(get_sequence_ids(tok_ids, self.tokenizer))
        return {"sequence_ids": sequence_ids}

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
        task_dataset = task_dataset.map(self.add_sequence_ids, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.generate_aux_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return task_dataset
