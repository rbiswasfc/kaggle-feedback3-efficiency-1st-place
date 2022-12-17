import os
import re
from copy import deepcopy

import numpy as np
import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer

#--------------- Tokenizer ---------------------------------------------#
NEW_TOKENS = [
    "[LF]",
    "[SOE]",
    "[EOE]",
]

TARGET_TOKENS = [
    "[=cohesion=]",
    "[=syntax=]",
    "[=vocabulary=]",
    "[=phraseology=]",
    "[=grammar=]",
    "[=conventions=]",
]

COHESION_TEXT = """Cohesion definition: Text organization consistently well controlled using a variety of effective \
    linguistic features such as reference and transitional words and  phrases to connect ideas \
    across sentences and  paragraphs; appropriate  overlap of ideas. Score: """

SYNTAX_TEXT = """Syntax definition: Flexible and effective use of a full range of syntactic structures including simple, \
    compound, and complex sentences; Negligible errors in sentence formation. Score: """

VOCABULARY_TEXT = """Vocabulary definition: Wide range of vocabulary flexibly and effectively used to convey precise  meanings; \
    skillful use  of topic-related terms and less common  words; rare negligible inaccuracies in word use. Score: """

PHRASEOLOGY_TEXT = """Phraseology definition: Flexible and effective use of a variety of phrases, such as idioms, collocations, \
    and  lexical bundles, to  convey precise and  subtle meanings; rare minor inaccuracies that are negligible. Score: """

GRAMMAR_TEXT = """Grammar definition: Command of grammar and  usage with few or no errors. Score: """

CONVENTIONS_TEXT = """Conventions definition: Consistent use of  appropriate conventions to convey meaning; spelling, \
    capitalization, and  punctuation errors  nonexistent or negligible. Score: """


# LABEL_INFO_MAP_OLD = {
#     "cohesion": "cohesion ",
#     "syntax": "syntax ",
#     "vocabulary": "vocabulary ",
#     "phraseology": "phraseology ",
#     "grammar": "grammar ",
#     "conventions": "conventions ",
# }

LABEL_INFO_MAP_OLD = {
    "cohesion": "cohesion (organization, transitional, overlap) ",
    "syntax": "syntax (formation, clauses) ",
    "vocabulary": "vocabulary (word variety) ",
    "phraseology": "phraseology (phrases, idioms, collocations, lexical bundles) ",
    "grammar": "grammar (morphology) ",
    "conventions": "conventions (spelling, capitalization, punctuation) ",
}

# LABEL_INFO_MAP = {
#     "cohesion": COHESION_TEXT,
#     "syntax": SYNTAX_TEXT,
#     "vocabulary": VOCABULARY_TEXT,
#     "phraseology": PHRASEOLOGY_TEXT,
#     "grammar": GRAMMAR_TEXT,
#     "conventions": CONVENTIONS_TEXT,
# }
LABEL_INFO_MAP_NEW0= {
    "cohesion": "Rate the following essay for cohesion (organization, transitional, overlap) score between 1 (bad) and 5 (great). The cohesion score for this text between 1 and 5 is the number ",
    "syntax": "Rate the following essay for syntax (formation, clauses) score between 1 (bad) and 5 (great). The syntax score for this text between 1 and 5 is the number ",
    "vocabulary": "Rate the following essay for vocabulary (word variety) score between 1 (bad) and 5 (great). The vocabulary score for this text between 1 and 5 is the number ",
    "phraseology": "Rate the following essay for phraseology (phrases, idioms, collocations, lexical bundles) score between 1 (bad) and 5 (great).The phraseology score for this text between 1 and 5 is the number ",
    "grammar": "Rate the following essay for grammar (morphology) score between 1 (bad) and 5 (great). The grammar score for this text between 1 and 5 is the number ",
    "conventions": "Rate the following essay for conventions (spelling, capitalization, punctuation) score between 1 (bad) and 5 (great). The conventions score for this text between 1 and 5 is the number ",
}
LABEL_INFO_MAP_NEW1 =  {
    "cohesion": "Rate the following essay for cohesion (organization, transitional, overlap) score between 1 (bad) and 5 (great). The cohesion score for this text between 1 and 5 is the number ",
    "syntax": "Rate the following essay for syntax (formation, clauses) score between 1 (bad) and 5 (great). The syntax score for this text between 1 and 5 is the number ",
    "vocabulary": "Rate the following essay for vocabulary (word variety) score between 1 (bad) and 5 (great). The vocabulary score for this text between 1 and 5 is the number ",
    "phraseology": "Rate the following essay for phraseology (phrases, idioms, collocations, lexical bundles) score between 1 (bad) and 5 (great).The phraseology score for this text between 1 and 5 is the number ",
    "grammar": "Rate the following essay for grammar (morphology) score between 1 (bad) and 5 (great). The grammar score for this text between 1 and 5 is the number ",
    "conventions": "Rate the following essay for conventions (spelling, capitalization, punctuation) score between 1 (bad) and 5 (great). The conventions score for this text between 1 and 5 is the number ",
}
LABEL_INFO_MAP_NEW2 = {
    "cohesion": 'Rate the following essay for cohesion (good text organization, transitional, overlap) with a score between 1 (bad) and 5 (great). The cohesion score for the following essay is 1 or 2 or 3 or 4 or 5? The score is ',
    "syntax": 'Rate the following essay for syntax (use of simple, compound and complex sentences with few errors) with a score between 1 (bad) and 5 (great). The syntax score for the following essay is 1 or 2 or 3 or 4 or 5? The score is ',
    "vocabulary": 'Rate the following essay for vocabulary (wide range of vocabulary) with a score between 1 (bad) and 5 (great). The vocabulary score for the following essay is 1 or 2 or 3 or 4 or 5? The score is ',
    "phraseology": 'Rate the following essay for phraseology (effective use of phrases, idioms, collocations, lexical bundles) with a score between 1 (bad) and 5 (great).The phraseology score for the following essay is 1 or 2 or 3 or 4 or 5? The score is ',
    "grammar": 'Rate the following essay for grammar (command of grammar with few or no errors) with a score between 1 (bad) and 5 (great). The grammar score for the following essay is 1 or 2 or 3 or 4 or 5? The score is ',
    "conventions": 'Rate the following essay for conventions (use of correct spelling, capitalization and punctuation with few errors) with a score between 1 (bad) and 5 (great). The conventions score for the following essay is 1 or 2 or 3 or 4 or 5? The score is ',
}

LABEL_INFO_MAP_NEW3 = {
    "cohesion": 'Rate the following essay for cohesion (good text organization, transitional, overlap). The cohesion score for the following essay is good or bad?. The score is ',
    "syntax": 'Rate the following essay for syntax (use of simple, compound and complex sentences with few errors). The syntax score for the following essay is good or bad?. The score is ',
    "vocabulary": 'Rate the following essay for vocabulary (wide range of vocabulary). The vocabulary score for the following essay is good or bad?. The score is ',
    "phraseology": 'Rate the following essay for phraseology (effective use of phrases, idioms, collocations, lexical bundles).The phraseology score for the following essay is good or bad?. The score is ',
    "grammar": 'Rate the following essay for grammar (command of grammar with few or no errors). The grammar score for the following essay is good or bad?. The score is ',
    "conventions": 'Rate the following essay for conventions (use of correct spelling, capitalization and punctuation with few errors). The conventions score for the following essay is good or bad?. The score is ',
}


#LABEL_INFO_MAP = LABEL_INFO_MAP_NEW #LABEL_INFO_MAP_OLD


def get_tokenizer(cfg):
    """load the tokenizer"""
    tokenizer_path = cfg.model.backbone_path
    print(f"loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # NEW TOKENS
    if cfg.model.add_new_tokens:
        print("adding new tokens...")
        tokens_to_add = []
        for this_tok in NEW_TOKENS:
            tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
        tokenizer.add_tokens(tokens_to_add)

    # TARGET TOKENS
    if cfg.model.add_target_tokens:
        print("adding target tokens...")
        tokens_to_add = []
        for this_tok in TARGET_TOKENS:
            tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
        tokenizer.add_tokens(tokens_to_add)

    print(f"tokenizer len: {len(tokenizer)}")

    test_string = "[SOE] This is a test \n [LF] [EOE] [=conventions=]!!"
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

        # additional tokens
        self.new_token_ids = set()
        self.target_token_ids = set(self.tokenizer.convert_tokens_to_ids([self.tokenizer.cls_token]))

        if self.cfg.model.add_new_tokens:
            self.new_token_ids = set(self.tokenizer.convert_tokens_to_ids(NEW_TOKENS))

        if self.cfg.model.add_target_tokens:
            self.target_token_ids = set(self.tokenizer.convert_tokens_to_ids(TARGET_TOKENS))

    def pre_process(self, df):
        df["full_text"] = df["full_text"].apply(lambda x: re.sub(re.compile(r'\n\n'), "[LF]", x))
        df["full_text"] = df["full_text"].apply(lambda x: " ".join(["[SOE]", x, "[EOE]"]))

        # add target wise prefix
        if self.cfg.model.add_target_tokens:
            prefix = [
                f"{LABEL_INFO_MAP['cohesion']} [=cohesion=]",
                f"{LABEL_INFO_MAP['syntax']} [=syntax=]",
                f"{LABEL_INFO_MAP['vocabulary']} [=vocabulary=]",
                f"{LABEL_INFO_MAP['phraseology']} [=phraseology=]",
                f"{LABEL_INFO_MAP['grammar']} [=grammar=]",
                f"{LABEL_INFO_MAP['conventions']} [=conventions=]",
            ]
            prefix = ", ".join(prefix)
            prefix += f"{self.tokenizer.sep_token}"
            df["full_text"] = df["full_text"].apply(lambda x: " ".join([prefix, x]))

        return df

    def tokenize_function(self, examples):
        tz = self.tokenizer(
            examples['full_text'],
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
            aux_labels.append(
                [
                    np.max(ex_labels),
                    np.min(ex_labels),
                    np.mean(ex_labels),
                ]
            )
        return {"aux_labels": aux_labels}

    def get_target_token_idxs(self, examples):
        target_token_idxs = []  # [] for _ in range(len(examples['input_ids']))]

        for example_input_ids in examples["input_ids"]:
            example_target_token_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id in self.target_token_ids]
            target_token_idxs.append(example_target_token_idxs)

        return {"target_token_idxs": target_token_idxs}

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
        print(f"{mode} dataframe sample:")
        print(df.head(1).T)
        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.get_target_token_idxs, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.generate_aux_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return task_dataset


#-------- Prompt based -----------------#

class FeedbackDatasetPrompt:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, cfg):
        # assign config
        self.cfg = cfg

        # label columns
        self.target_names = cfg.model.target_names

        # load tokenizer
        self.load_tokenizer()

        self.pos_tok = "good"
        self.neg_tok = "bad"

        self.pos_tok_id = self.tokenizer.convert_tokens_to_ids(self.pos_tok)
        self.neg_tok_id = self.tokenizer.convert_tokens_to_ids(self.neg_tok)

        print(f"positive token: {self.pos_tok}, id: {self.pos_tok_id}")
        print(f"negative token: {self.neg_tok}, id: {self.neg_tok_id}")

        if self.cfg.model.prompt_type == 'pv':
            self.pos_tok_list = ["4", "four", "Four", "5", "five", "Five"]
            self.neg_tok_list = ["1", "one", "One", "2", "two", "Two"]
        elif self.cfg.model.prompt_type == 'pv1':
            self.pos_tok_list = ["4", "four", "Four", "5", "five", "Five", "3", "three", "Three"]
            self.neg_tok_list = ["1", "one", "One", "2", "two", "Two", "0", "zero", "Zero"]
        elif self.cfg.model.prompt_type == 'pv2':
            self.pos_tok_list = ["3", "4", "5",]
            self.neg_tok_list = ["1", "2"]
        elif self.cfg.model.prompt_type == 'pv3':
            #self.pos_tok_list = ["good", "great"]  # "acceptable", "excellent"
            #self.neg_tok_list = ["bad", "poor"]  # "unacceptable", "terrible"
            self.pos_tok_list = ["good"]
            self.neg_tok_list = ["bad"]
        else:
            self.pos_tok_list = ["good", "fine"]  # "acceptable", "excellent"
            self.neg_tok_list = ["bad", "poor"]   # "unacceptable", "terrible"


        self.pos_tok_id_list = self.tokenizer.convert_tokens_to_ids(self.pos_tok_list)
        self.neg_tok_id_list = self.tokenizer.convert_tokens_to_ids(self.neg_tok_list)

        print(f"positive token list: {self.pos_tok_list}, id: {self.pos_tok_id_list}")
        print(f"negative token list: {self.neg_tok_list}, id: {self.neg_tok_id_list}")

    def load_tokenizer(self):
        self.tokenizer = get_tokenizer(self.cfg)

        # additional tokens
        self.new_token_ids = set()
        if self.cfg.model.add_new_tokens:
            self.new_token_ids = set(self.tokenizer.convert_tokens_to_ids(NEW_TOKENS))

    def pre_process(self, df):
        df["full_text"] = df["full_text"].apply(lambda x: re.sub(re.compile(r'\n\n'), "[LF]", x))
        df["full_text"] = df["full_text"].apply(lambda x: " ".join(["[SOE]", x, "[EOE]"]))

        if self.cfg.model.prompt_type == 'new':
            LABEL_INFO_MAP = LABEL_INFO_MAP_NEW0
        elif self.cfg.model.prompt_type == 'pv1':
            LABEL_INFO_MAP = LABEL_INFO_MAP_NEW1
        elif self.cfg.model.prompt_type == 'pv2':
            LABEL_INFO_MAP = LABEL_INFO_MAP_NEW2
        elif self.cfg.model.prompt_type == 'pv3':
            LABEL_INFO_MAP = LABEL_INFO_MAP_NEW3
        else:
            LABEL_INFO_MAP = LABEL_INFO_MAP_OLD

        # add target wise prefix
        definitions = [
            f"{LABEL_INFO_MAP['cohesion']}",
            f"{LABEL_INFO_MAP['syntax']}",
            f"{LABEL_INFO_MAP['vocabulary']}",
            f"{LABEL_INFO_MAP['phraseology']}",
            f"{LABEL_INFO_MAP['grammar']}",
            f"{LABEL_INFO_MAP['conventions']}",
        ]


        if 'new' in self.cfg.model.prompt_type:
            prefix =  f"{self.tokenizer.mask_token}. ".join(definitions)
            prefix += f"{self.tokenizer.mask_token}. {self.tokenizer.sep_token}"
        else:
            prefix = "Evaluate the following essay based on " + f"{self.tokenizer.mask_token}, ".join(definitions)
            prefix += f"{self.tokenizer.mask_token}. {self.tokenizer.sep_token}"


        # prefix = f"The following essay has {self.tokenizer.mask_token} cohesion, {self.tokenizer.mask_token} syntax, {self.tokenizer.mask_token} vocabulary, {self.tokenizer.mask_token} phraseology, {self.tokenizer.mask_token} grammar and {self.tokenizer.mask_token} conventions."
        # prefix += f"{self.tokenizer.sep_token}"

        df["full_text"] = df["full_text"].apply(lambda x: " ".join([prefix, x]))

        print("sample text:")
        print(df.sample().full_text.values[0])
        with open (os.path.join(self.cfg.outputs.model_dir, 'pv.txt'), 'w') as f:
            f.write(f'Pos tok id: {self.pos_tok_list}\n')
            f.write(f'Neg tok id: {self.neg_tok_list}\n')
            f.write(df.sample().full_text.values[0])

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
            aux_labels.append(
                [
                    np.max(ex_labels),
                    np.min(ex_labels),
                    np.mean(ex_labels),
                ]
            )
        return {"aux_labels": aux_labels}

    def get_target_token_idxs(self, examples):
        target_token_idxs = []
        mask_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        for example_input_ids in examples["input_ids"]:
            example_target_token_idxs = [pos for pos, this_id in enumerate(
                example_input_ids) if this_id == mask_token_id]
            target_token_idxs.append(example_target_token_idxs)

        return {"target_token_idxs": target_token_idxs}

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
        print(f"{mode} dataframe sample:")
        print(df.head(1).T)
        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.get_target_token_idxs, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
            task_dataset = task_dataset.map(self.generate_aux_labels, batched=True)

        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return task_dataset
