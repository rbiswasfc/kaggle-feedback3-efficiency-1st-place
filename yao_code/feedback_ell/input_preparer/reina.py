"""
@created by: heyao
@created at: 2022-10-25 14:34:11
"""
import heapq
from typing import List, Union

import numpy as np
import transformers
from tqdm.auto import tqdm
from nltk import word_tokenize
from omegaconf import DictConfig
from rank_bm25 import BM25Okapi
import pandas as pd

from feedback_ell.input_preparer import BaseInputPreparer
from feedback_ell.utils import label_columns


class ReinaIndexer(object):
    def __init__(self, cut_method="split"):
        self.cut_method = cut_method
        self.indexer: Union[BM25Okapi, None] = None

    def cut(self):
        if self.cut_method == "split":
            return lambda x: x.split()
        if self.cut_method == "nltk":
            return word_tokenize
        raise ValueError(f"invalid cut method: {self.cut_method}")

    def fit(self, corpus):
        self.indexer = BM25Okapi(corpus)
        return self

    def index(self, query: List[str], remove_self=True):
        scores = self.indexer.get_scores(query)
        if not remove_self:
            return int(scores.argmax())
        indexes = heapq.nlargest(2, range(len(scores)), scores.take)
        return indexes[-1]


class ReinaInputPreparer(BaseInputPreparer):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        super().__init__(tokenizer, config)
        self.indexer = None

    def prepare_input(self, df: pd.DataFrame, df_train: Union[pd.DataFrame, None] = None) -> [List, List, List]:
        # ===== don't drop index =====
        strip_text = self.config.dataset.get("strip_text", True)
        max_length = self.config.train.max_length
        if strip_text:
            texts = df["full_text"].str.strip().to_list()
            df_train["full_text"] = df_train["full_text"].str.strip().to_list()
        else:
            texts = df["full_text"].to_list()
            # train_texts = df_train["full_text"].to_list()
        if self.indexer is None:
            self.indexer = ReinaIndexer()
            self.indexer.fit([i.split() for i in df_train["full_text"].to_list()])

        is_train = self.config.train.get("is_train", True)
        # fold = self.config.train.fold_index[0]
        most_similar_indexes = np.load(self.config.dataset.train_bm25_path)
        df_most_similar_indexes = pd.DataFrame(most_similar_indexes)
        # reset diag and val_idx's BM25 score to 0.
        val_idx = ~df_most_similar_indexes.index.isin(df_train.index)
        df_most_similar_indexes.loc[:, val_idx] = 0
        if is_train:
            _arr = df_most_similar_indexes.values
            _arr[range(len(_arr)), range(len(_arr))] = 0
            df_most_similar_indexes = pd.DataFrame(_arr)

        # get most similar index from `df` (in df_train) This is a index in the full data.
        most_similar_samples = df_most_similar_indexes.values.argmax(axis=1).tolist()  # [FULL_DATA_SIZE, 1]
        # print(len(most_similar_samples), most_similar_samples)

        df_most_similar = df_train.loc[most_similar_samples].reset_index(drop=True).loc[df.index]
        # print(df_most_similar.shape)
        if self.label_columns[0] in df.columns:
            if self.config.train.multi_task.enable:
                raise RuntimeError("Retrieval-based method didn't support multi task now.")
                # labels = df[self.label_columns + self.config.train.multi_task.get("tasks", [])].values
            else:
                if is_train:
                    gt = df_train[self.label_columns].values
                    train_labels = df_most_similar[self.label_columns].values
                    # print(df_train.index, df_most_similar.index)
                else:
                    gt = df[self.label_columns].values
                    train_labels = df_most_similar[self.label_columns].values
                    # print(df.index, df_most_similar.index)
                target_labels = train_labels.tolist()
                # labels = (gt - train_labels).tolist()
                labels = gt.tolist()
        else:
            labels = []
            target_labels = df_train[self.label_columns].values.tolist()
        print(labels[0])
        print(target_labels[0])
        encodings = []
        cls_token_id = [self.tokenizer.cls_token_id]
        sep_token_id = [self.tokenizer.sep_token_id]
        # original_text, retried_text
        for i, (text, train_anchor, label_anchor) in tqdm(enumerate(zip(texts, df_most_similar.full_text.to_list(), target_labels))):
            encoding_target = self.tokenizer(text, max_length=max_length, padding=False, truncation=True,
                                             add_special_tokens=False, return_token_type_ids=False)
            input_text = f"label: {', '.join(f'{self.label_columns[i]}({label_anchor[i]})' for i in range(6))}{self.tokenizer.sep_token}" + train_anchor
            encoding_anchor = self.tokenizer(input_text, max_length=max_length, padding=False, truncation=True,
                                             add_special_tokens=False, return_token_type_ids=False)
            input_ids = cls_token_id + encoding_target["input_ids"] + sep_token_id + \
                        encoding_anchor["input_ids"] + sep_token_id
            encoding = {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids)
            }
            encodings.append(encoding)
        return encodings, labels, target_labels, df["text_id"].to_list()


if __name__ == '__main__':
    from collections import Counter
    from itertools import chain

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    from feedback_ell.utils import label_columns

    pd.options.display.max_columns = 100

    f = "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv"
    df = pd.read_csv(f)
    indexer = ReinaIndexer()
    # corpus = [
    #     ["I", "am", "ok"],
    #     ["are", "you", "ok"],
    #     ["It", "is", "good"]
    # ]
    # corpus = df.full_text.str.strip().str.split().to_list()
    # indexer.fit(corpus)
    filename = "/home/heyao/kaggle/feedback-ells/input/all_data_bm25.npz.npy"
    # print(f"generate similar samples to cache: {filename}")
    # most_similar_samples = []
    # for words in tqdm(corpus):
    #     # words = text.split()
    #     most_similar_samples.append(indexer.indexer.get_scores(words).tolist())
    # np.save(filename, np.array(most_similar_samples))

    # most_similar_samples = np.load(filename)
    # print(most_similar_samples.shape)
    # print(most_similar_samples[0])
    #
    # # best_match = indexer.index(corpus[0], remove_self=True)
    # # print(best_match, type(best_match))
    # df_train = df.loc[:3000]
    # df_val = df.loc[3000:]
    df_train = df.sample(3000)
    df_val = df.loc[~df.index.isin(df_train.index)]
    # train_idx = df_train.index.to_list()
    # val_idx = df_val.index.to_list()
    # df_most_similar = pd.DataFrame(most_similar_samples)
    # print(val_idx)
    # print(df_most_similar.loc[val_idx, train_idx].shape)
    # print(df_most_similar.loc[val_idx].shape)
    # df_most_similar.loc[val_idx, val_idx] = 0
    # # val_idx = ~df.index.isin(df_train.index)
    # # print(df_most_similar.loc[val_idx, val_idx])
    # print(df_most_similar.loc[val_idx].values.argmax(axis=1).shape)
    # print(df_most_similar.loc[val_idx].values.argmax(axis=1)[:10])

    # ================== SEP ================= #
    config = OmegaConf.load("../../config/deberta_v3_large_reg.yaml")
    # folds = pd.read_csv(config.dataset.fold_file)
    # for fold in range(4):
    #     pass

    model_path = "/media/heyao/42f00068-ba8d-48dd-8719-61eda5244b8d/pretrained-models/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    input_preparer = ReinaInputPreparer(tokenizer, config)
    train_inputs, train_labels, train_target_labels, train_ids = input_preparer.prepare_input(df_train, df_train)
    input_preparer.config.train.is_train = False  # !!! important !!!
    val_inputs, val_labels, val_target_labels, val_ids = input_preparer.prepare_input(df_val, df_train)
    print(len(train_inputs), len(train_labels), len(train_target_labels), len(train_ids))
    print(len(val_inputs), len(val_labels), len(val_target_labels), len(val_ids))
    print(train_inputs[0])
    print(train_labels[0])
    print("=" * 60)
    print(val_inputs[0])
    print(val_labels[:3])
    print(tokenizer.decode(train_inputs[0]["input_ids"]))
    print(tokenizer.decode(val_inputs[0]["input_ids"]))
    print(max([len(i["input_ids"]) for i in train_inputs]))
    print(max([len(i["input_ids"]) for i in val_inputs]))
    # sns.distplot([i[0] for i in train_labels], label=f"{label_columns[0]}-diff")
    # sns.distplot([i for i in df_train[label_columns[0]]], label=f"{label_columns[0]}")
    # sns.distplot([i[0] for i in val_labels], label=f"val-{label_columns[0]}-diff")
    # sns.distplot([i for i in df_val[label_columns[0]]], label=f"val-{label_columns[0]}")
    # plt.legend()
    # plt.show()
    df_labels = pd.DataFrame(train_labels)
    df_labels.columns = label_columns
    print(df_labels.describe())
    df_labels = pd.DataFrame(val_labels)
    df_labels.columns = label_columns
    print(df_labels.describe())
