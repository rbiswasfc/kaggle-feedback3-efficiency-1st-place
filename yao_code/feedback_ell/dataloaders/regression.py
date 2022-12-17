"""
@created by: heyao
@created at: 2022-09-05 00:14:53
"""
import pandas as pd
import torch

from feedback_ell.input_preparer.external_tags import ExternalTagRegressionInputPreparer
from feedback_ell.input_preparer.one_target import OneTargetRegressionInputPreparer
from feedback_ell.input_preparer.regression import NoParagraphOrderRegressionInputPreparer
from feedback_ell.utils.aug.rand_shuffle import shuffle_paragraph
from feedback_ell.utils.dataset.simple import CompetitionDataset, AddMaskTaskDataset
from feedback_ell.input_preparer import RegressionInputPreparer
from feedback_ell.utils.collators.sequence_bucket import SequenceBucketPadCollator, MaxSeqLenPadCollator


def _make_dataloader_from_encodings(encodings, labels, tokenizer, config, is_train, shuffle, pad_to_batch=True,
                                    mask_ratio=0.0, weights=None):
    batch_size = config.train.batch_size
    num_workers = config.train.num_workers
    collate_class = SequenceBucketPadCollator if pad_to_batch else MaxSeqLenPadCollator
    collate_fn = collate_class(config.train.max_length + 256, tokenizer, is_train=is_train, target_is_float=True,
                               pad_val=-100)
    dataset_class = CompetitionDataset
    if config.train.get("do_mlm", False) and shuffle:
        dataset_class = AddMaskTaskDataset
    print(f"use {dataset_class} as input")
    dataset = dataset_class(tokenizer=tokenizer, x=encodings, y=labels, weights=weights,
                            mask_ratio=mask_ratio, reweight=None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                             collate_fn=collate_fn)
    return dataloader


def make_dataloader(df, tokenizer, config, folds=None, fold=None, sort_val_set=False, pad_to_batch=True,
                    mask_ratio=0.0):
    if fold is not None and folds is None:
        # no fold info to get train, eval set, raise error
        raise ValueError("folds and fold can't both be None")
    if config.train.get("tag_task", {}).get("enable", False):
        input_preparer = ExternalTagRegressionInputPreparer(tokenizer, config)
    else:
        input_preparer = RegressionInputPreparer(tokenizer, config)
    if fold is None:
        # return full dataloader
        if "weight" not in df.columns:
            weights = None
        else:
            weights = df["weight"].tolist()
        encodings, labels, text_ids = input_preparer.prepare_input(df)
        dataloader = _make_dataloader_from_encodings(encodings, labels.tolist(), tokenizer, config,
                                                     is_train=True, shuffle=False, pad_to_batch=pad_to_batch,
                                                     mask_ratio=mask_ratio, weights=weights)
        return dataloader, text_ids
    df_train = df[folds.kfold != fold].reset_index(drop=True)
    train_encodings, train_labels, train_text_ids = input_preparer.prepare_input(df_train)
    train_loader = _make_dataloader_from_encodings(train_encodings, train_labels.tolist(), tokenizer, config,
                                                   is_train=True, shuffle=True, pad_to_batch=pad_to_batch,
                                                   mask_ratio=mask_ratio)

    df_val = df[folds.kfold == fold].reset_index(drop=True)
    val_encodings, val_labels, val_text_ids = input_preparer.prepare_input(df_val)
    val_labels = val_labels.tolist()
    # sort val data
    if sort_val_set:
        lengths = [sum(i["attention_mask"]) for i in val_encodings]
        sorted_values = sorted(zip(val_encodings, val_labels, val_text_ids, lengths), key=lambda x: x[-1])
        val_encodings, val_labels, val_text_ids = [], [], []
        for e, l, t, _ in sorted_values:
            val_encodings.append(e)
            val_labels.append(l)
            val_text_ids.append(t)
    val_loader = _make_dataloader_from_encodings(val_encodings, val_labels, tokenizer, config,
                                                 is_train=True, shuffle=False, pad_to_batch=pad_to_batch,
                                                 mask_ratio=0)
    return (train_loader, train_text_ids), (val_loader, val_text_ids)


def make_dataloader_stage1(df, tokenizer, config, folds=None, fold=None, sort_val_set=False,
                           pad_to_batch=True, mask_ratio=0.0, shuffle=False):
    assert folds is None and fold is None

    input_preparer = RegressionInputPreparer(tokenizer, config)
    encodings, labels, text_ids = input_preparer.prepare_input(df)
    if sort_val_set:
        val_encodings, val_labels, val_text_ids = [], [], []
        lengths = [sum(i["attention_mask"]) for i in encodings]
        sorted_values = sorted(zip(encodings, labels, text_ids, lengths), key=lambda x: x[-1])
        for e, l, t, _ in sorted_values:
            val_encodings.append(e)
            val_labels.append(l.tolist())
            val_text_ids.append(t)
    else:
        val_encodings = encodings
        val_labels = [i.tolist() for i in labels]
        val_text_ids = text_ids
    dataloader = _make_dataloader_from_encodings(val_encodings, val_labels, tokenizer, config,
                                                 is_train=True, shuffle=shuffle, pad_to_batch=pad_to_batch,
                                                 mask_ratio=mask_ratio)
    return dataloader, val_text_ids


if __name__ == '__main__':
    import os

    from transformers import AutoTokenizer
    from omegaconf import OmegaConf

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = OmegaConf.load("../../config/deberta_v3_base_reg.yaml")
    df = pd.read_csv("../../input/feedback-prize-english-language-learning/train.csv")
    folds = pd.read_csv("../../input/fb3_folds.csv")
    fold = None
    tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    config.train.do_mlm = True

    # (dataloader, text_ids), (val_loader, val_ids) = make_dataloader(df, tokenizer, config, folds=folds,
    #                                                                 fold=fold, sort_val_set=True)
    dataloader, text_ids = make_dataloader(df, tokenizer, config, folds=folds, fold=fold)
    x, y = next(iter(dataloader))
    for key in x.keys():
        print(key, x[key].shape)
    print(y.shape)
    # for x, _ in val_loader:
    #     print(x["attention_mask"].sum(axis=1))
