import os
import re

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
        if self.cfg.model.add_new_tokens:
            self.new_token_ids = set(self.tokenizer.convert_tokens_to_ids(NEW_TOKENS))

    def pre_process(self, df):
        df["full_text"] = df["full_text"].apply(lambda x: re.sub(re.compile(r'\n\n'), " [LF] ", x))
        df["full_text"] = df["full_text"].apply(lambda x: " ".join(["[SOE]", x, "[EOE]"]))
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
        print(f"{mode} dataframe sample:")
        print(df.head(1).T)
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

#----------- Dataset Pairwise -------------------------------------------------------------#


# def prepare_pretraining_data(input_df, mode="cos_sim", n_samples=16):
#     """prepares training data for sentence transformer
#     cos_sim mode: cosine similarity
#     contrastive mode: contrastive learning
#     """

#     # sampling
#     setfit_text_ids = set()
#     targets = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
#     scores = [1.0, 5.0, 2.0, 4.0, 3.0]

#     input_df = input_df.sample(frac=1.0)

#     for t in targets:
#         for s in scores:
#             if (s < 1.1) | (s > 4.9):
#                 candidate_df = input_df[input_df[t] == s].copy()
#                 for this_id in candidate_df["text_id"].tolist():
#                     setfit_text_ids.add(this_id)
#             else:
#                 bank_df = input_df[input_df["text_id"].isin(setfit_text_ids)].copy()
#                 n_required = max(0, n_samples - len(bank_df[bank_df[t] == s]))
#                 candidate_df = input_df[input_df[t] == s].copy()
#                 if n_required > 0:
#                     n_required = min(n_required, len(candidate_df))
#                     candidate_df["range"] = candidate_df[targets].apply(lambda x: max(x)-min(x), axis=1)
#                     candidate_df = candidate_df.sort_values(by='range', ascending=False)
#                     tmp = candidate_df.head(n_required)  # candidate_df.sample(n_required)
#                     print(tmp.head(1).T)
#                     for this_id in tmp["text_id"].tolist():
#                         setfit_text_ids.add(this_id)

#     focus_df = input_df[input_df["text_id"].isin(setfit_text_ids)].copy()
#     focus_df = focus_df.reset_index(drop=True)
#     print(f"shape of selected data: {focus_df.shape}")

#     def contrast_rater(x):
#         thres = 0.01
#         if abs(x) < thres:
#             return 1
#         else:
#             return 0

#     # pre-process scores
#     def scale_score(x):
#         x = (x-1.0)/(4.0)
#         return x

#     target_cols = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
#     for col in target_cols:
#         focus_df[col] = focus_df[col].apply(scale_score)

#     # create pairwise data
#     data = []
#     stack = set()

#     for _, row in focus_df.iterrows():
#         text_id_1 = row.text_id
#         stack.add(text_id_1)

#         full_text_1 = row.full_text
#         cohesion_1 = row.cohesion
#         syntax_1 = row.syntax
#         vocabulary_1 = row.vocabulary
#         phraseology_1 = row.phraseology
#         grammar_1 = row.grammar
#         conventions_1 = row.conventions

#         for _, tmp in focus_df.iterrows():
#             text_id_2 = tmp.text_id
#             if text_id_2 in stack:
#                 continue
#             full_text_2 = tmp.full_text
#             cohesion_2 = tmp.cohesion
#             syntax_2 = tmp.syntax
#             vocabulary_2 = tmp.vocabulary
#             phraseology_2 = tmp.phraseology
#             grammar_2 = tmp.grammar
#             conventions_2 = tmp.conventions

#             # -------
#             if mode == "cos_sim":
#                 data.append(
#                     [
#                         text_id_1,
#                         full_text_1,
#                         text_id_2,
#                         full_text_2,

#                         1.0 - 2.0*abs(cohesion_1 - cohesion_2),
#                         1.0 - 2.0*abs(syntax_1 - syntax_2),
#                         1.0 - 2.0*abs(vocabulary_1 - vocabulary_2),
#                         1.0 - 2.0*abs(phraseology_1 - phraseology_2),
#                         1.0 - 2.0*abs(grammar_1 - grammar_2),
#                         1.0 - 2.0*abs(conventions_1 - conventions_2),
#                     ]
#                 )
#             elif mode == "contrastive":
#                 data.append(
#                     [
#                         text_id_1,
#                         full_text_1,
#                         text_id_2,
#                         full_text_2,

#                         contrast_rater(cohesion_1 - cohesion_2),
#                         contrast_rater(syntax_1 - syntax_2),
#                         contrast_rater(vocabulary_1 - vocabulary_2),
#                         contrast_rater(phraseology_1 - phraseology_2),
#                         contrast_rater(grammar_1 - grammar_2),
#                         contrast_rater(conventions_1 - conventions_2),
#                     ]
#                 )
#             else:
#                 raise NotImplementedError
#         # -----------
#     pair_df = pd.DataFrame(data)
#     pair_df.columns = ["text_id_sent_1", "full_text_sent_1", "text_id_sent_2", "full_text_sent_2",
#                        "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

#     print("pair df stats: ")
#     print(pair_df.describe())

#     print(f"shape of pairwise data: {pair_df.shape}")
#     print("cohesion value counts:")
#     print(focus_df["cohesion"].value_counts().sort_index())
#     return pair_df

def prepare_pretraining_data(input_df, mode="cos_sim", n_samples=16):
    """prepares training data for sentence transformer
    cos_sim mode: cosine similarity
    contrastive mode: contrastive learning
    """

    # sampling
    setfit_text_ids = set()
    targets = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]

    input_df = input_df.sample(frac=1.0)

    for t in targets:
        for s in scores:
            bank_df = input_df[input_df["text_id"].isin(setfit_text_ids)].copy()
            n_required = max(0, n_samples - len(bank_df[bank_df[t] == s]))
            candidate_df = input_df[input_df[t] == s].copy()
            if n_required > 0:
                n_required = min(n_required, len(candidate_df))
                tmp = candidate_df.sample(n_required)
                for this_id in tmp["text_id"].tolist():
                    setfit_text_ids.add(this_id)

    focus_df = input_df[input_df["text_id"].isin(setfit_text_ids)].copy()
    focus_df = focus_df.reset_index(drop=True)
    print(f"shape of selected data: {focus_df.shape}")

    def contrast_rater(x):
        thres = 0.01
        if abs(x) < thres:
            return 1
        else:
            return 0

    # pre-process scores
    def scale_score(x):
        x = (x-1.0)/(4.0)
        return x

    target_cols = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    for col in target_cols:
        focus_df[col] = focus_df[col].apply(scale_score)

    # create pairwise data
    data = []
    stack = set()

    for _, row in focus_df.iterrows():
        text_id_1 = row.text_id
        stack.add(text_id_1)

        full_text_1 = row.full_text
        cohesion_1 = row.cohesion
        syntax_1 = row.syntax
        vocabulary_1 = row.vocabulary
        phraseology_1 = row.phraseology
        grammar_1 = row.grammar
        conventions_1 = row.conventions

        for _, tmp in focus_df.iterrows():
            text_id_2 = tmp.text_id
            if text_id_2 in stack:
                continue
            full_text_2 = tmp.full_text
            cohesion_2 = tmp.cohesion
            syntax_2 = tmp.syntax
            vocabulary_2 = tmp.vocabulary
            phraseology_2 = tmp.phraseology
            grammar_2 = tmp.grammar
            conventions_2 = tmp.conventions

            # -------
            if mode == "cos_sim":
                data.append(
                    [
                        text_id_1,
                        full_text_1,
                        text_id_2,
                        full_text_2,

                        1.0 - 2.0*abs(cohesion_1 - cohesion_2),
                        1.0 - 2.0*abs(syntax_1 - syntax_2),
                        1.0 - 2.0*abs(vocabulary_1 - vocabulary_2),
                        1.0 - 2.0*abs(phraseology_1 - phraseology_2),
                        1.0 - 2.0*abs(grammar_1 - grammar_2),
                        1.0 - 2.0*abs(conventions_1 - conventions_2),
                    ]
                )
            elif mode == "contrastive":
                data.append(
                    [
                        text_id_1,
                        full_text_1,
                        text_id_2,
                        full_text_2,

                        contrast_rater(cohesion_1 - cohesion_2),
                        contrast_rater(syntax_1 - syntax_2),
                        contrast_rater(vocabulary_1 - vocabulary_2),
                        contrast_rater(phraseology_1 - phraseology_2),
                        contrast_rater(grammar_1 - grammar_2),
                        contrast_rater(conventions_1 - conventions_2),
                    ]
                )
            else:
                raise NotImplementedError
        # -----------
    pair_df = pd.DataFrame(data)
    pair_df.columns = ["text_id_sent_1", "full_text_sent_1", "text_id_sent_2", "full_text_sent_2",
                       "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    print("pair df stats: ")
    print(pair_df.describe())

    print(f"shape of pairwise data: {pair_df.shape}")
    print("cohesion value counts:")
    print(focus_df["cohesion"].value_counts().sort_index())
    return pair_df

# def prepare_pretraining_data(input_df, pair_factor, mode="cos_sim"):
#     """prepares training data for sentence transformer
#     cos_sim mode: cosine similarity
#     contrastive mode: contrastive learning
#     """
#     def contrast_rater(x):
#         thres = 0.01
#         if abs(x) < thres:
#             return 1
#         else:
#             return 0

#     # pre-process scores
#     def scale_score(x):
#         x = (x-1.0)/(4.0)
#         return x
#     target_cols = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
#     for col in target_cols:
#         input_df[col] = input_df[col].apply(scale_score)

#     # create pairwise data
#     data = []
#     for _, row in input_df.iterrows():
#         text_id_1 = row.text_id
#         full_text_1 = row.full_text

#         cohesion_1 = row.cohesion
#         syntax_1 = row.syntax
#         vocabulary_1 = row.vocabulary
#         phraseology_1 = row.phraseology
#         grammar_1 = row.grammar
#         conventions_1 = row.conventions

#         for _ in range(pair_factor):
#             tmp = input_df.sample()
#             text_id_2 = tmp.text_id.values[0]
#             full_text_2 = tmp.full_text.values[0]

#             cohesion_2 = tmp.cohesion.values[0]
#             syntax_2 = tmp.syntax.values[0]
#             vocabulary_2 = tmp.vocabulary.values[0]
#             phraseology_2 = tmp.phraseology.values[0]
#             grammar_2 = tmp.grammar.values[0]
#             conventions_2 = tmp.conventions.values[0]

#             # -------
#             if mode == "cos_sim":
#                 data.append(
#                     [
#                         text_id_1,
#                         full_text_1,
#                         text_id_2,
#                         full_text_2,
#                         # (cohesion_1 - cohesion_2),
#                         # (syntax_1 - syntax_2),
#                         # (vocabulary_1 - vocabulary_2),
#                         # (phraseology_1 - phraseology_2),
#                         # (grammar_1 - grammar_2),
#                         # (conventions_1 - conventions_2),

#                         1.0 - 2.0*abs(cohesion_1 - cohesion_2),
#                         1.0 - 2.0*abs(syntax_1 - syntax_2),
#                         1.0 - 2.0*abs(vocabulary_1 - vocabulary_2),
#                         1.0 - 2.0*abs(phraseology_1 - phraseology_2),
#                         1.0 - 2.0*abs(grammar_1 - grammar_2),
#                         1.0 - 2.0*abs(conventions_1 - conventions_2),
#                     ]
#                 )
#             elif mode == "contrastive":
#                 data.append(
#                     [
#                         text_id_1,
#                         full_text_1,
#                         text_id_2,
#                         full_text_2,

#                         contrast_rater(cohesion_1 - cohesion_2),
#                         contrast_rater(syntax_1 - syntax_2),
#                         contrast_rater(vocabulary_1 - vocabulary_2),
#                         contrast_rater(phraseology_1 - phraseology_2),
#                         contrast_rater(grammar_1 - grammar_2),
#                         contrast_rater(conventions_1 - conventions_2),
#                     ]
#                 )
#             else:
#                 raise NotImplementedError
#         # -----------
#     pair_df = pd.DataFrame(data)
#     pair_df.columns = ["text_id_sent_1", "full_text_sent_1", "text_id_sent_2", "full_text_sent_2",
#                        "cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

#     print("pair df stats: ")
#     print(pair_df.describe())
#     return pair_df


class FeedbackDatasetPairwise:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, cfg):
        # assign config
        self.cfg = cfg

        # label columns
        # self.target_names = [f"{tn}_diff" for tn in cfg.model.target_names]
        self.target_names = cfg.model.target_names

        # load tokenizer
        self.load_tokenizer()

    def load_tokenizer(self):
        self.tokenizer = get_tokenizer(self.cfg)

        # additional tokens
        self.new_token_ids = set()
        if self.cfg.model.add_new_tokens:
            self.new_token_ids = set(self.tokenizer.convert_tokens_to_ids(NEW_TOKENS))

    def pre_process(self, df):
        df["full_text"] = df["full_text"].apply(lambda x: re.sub(re.compile(r'\n\n'), " [LF] ", x))
        df["full_text"] = df["full_text"].apply(lambda x: " ".join(["[SOE]", x, "[EOE]"]))
        return df

    def tokenize_function_sent_1(self, examples):
        tz = self.tokenizer(
            examples["full_text_sent_1"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
        )
        to_return = {
            "input_ids_sent_1": tz["input_ids"],
            "attention_mask_sent_1": tz["attention_mask"],
        }
        return to_return

    def tokenize_function_sent_2(self, examples):
        tz = self.tokenizer(
            examples["full_text_sent_2"],
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            add_special_tokens=True,
        )
        to_return = {
            "input_ids_sent_2": tz["input_ids"],
            "attention_mask_sent_2": tz["attention_mask"],
        }
        return to_return

    def generate_labels(self, examples):
        labels = [[] for _ in range(len(examples['full_text_sent_2']))]
        for col in self.target_names:
            for i, val in enumerate(examples[col]):
                labels[i].append(val)
        return {"labels": labels}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) + len(y) for x, y in zip(examples["input_ids_sent_1"], examples["input_ids_sent_2"])]}

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

        if mode == "train":
            pair_df = prepare_pretraining_data(df, self.cfg.model.label_mode, n_samples=64)
        else:
            pair_df = prepare_pretraining_data(df, self.cfg.model.label_mode, n_samples=5)

        print(f"{mode} dataframe sample:")
        print(pair_df.head(1).T)
        task_dataset = Dataset.from_pandas(pair_df)
        task_dataset = task_dataset.map(self.tokenize_function_sent_1, batched=True)
        task_dataset = task_dataset.map(self.tokenize_function_sent_2, batched=True)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            print(e)
        return task_dataset
