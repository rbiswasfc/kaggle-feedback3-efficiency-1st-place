"""
@created by: heyao
@created at: 2022-09-05 00:14:53
"""
import pandas as pd
import torch

from feedback_ell.input_preparer.external_tags import ExternalTagRegressionInputPreparer
from feedback_ell.input_preparer.reina import ReinaInputPreparer
from feedback_ell.utils.dataset.simple import CompetitionDataset, AddMaskTaskDataset, ReinaDataset
from feedback_ell.input_preparer import RegressionInputPreparer
from feedback_ell.utils.collators.sequence_bucket import ReinaSequenceBucketPadCollator


def _make_dataloader_from_encodings(encodings, labels, target_labels, tokenizer, config, is_train, shuffle,
                                    pad_to_batch=True, mask_ratio=0.0):
    batch_size = config.train.batch_size
    num_workers = config.train.num_workers
    collate_class = ReinaSequenceBucketPadCollator
    collate_fn = collate_class(config.train.max_length * 2 + 3, tokenizer, is_train=is_train, target_is_float=True,
                               pad_val=-100)
    dataset_class = ReinaDataset
    print(f"use {dataset_class} as input")
    # labels and target_labels is inverse
    dataset = dataset_class(x=encodings, y_target=target_labels, y=labels, mask_ratio=mask_ratio, tokenizer=tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                             collate_fn=collate_fn)
    return dataloader


def make_dataloader(df, df_train, tokenizer, config, folds=None, fold=None, sort_val_set=False, pad_to_batch=True,
                    mask_ratio=0.0):
    if fold is not None and folds is None:
        # no fold info to get train, eval set, raise error
        raise ValueError("folds and fold can't both be None")
    input_preparer = ReinaInputPreparer(tokenizer, config)
    if fold is None:
        # return full dataloader
        input_preparer.config.train.is_train = False
        encodings, labels, target_labels, text_ids = input_preparer.prepare_input(df, df_train=df_train)
        dataloader = _make_dataloader_from_encodings(encodings, labels.tolist(), target_labels.tolist(),
                                                     tokenizer, config,
                                                     is_train=True, shuffle=False, pad_to_batch=pad_to_batch,
                                                     mask_ratio=mask_ratio)
        return dataloader, text_ids
    # return train_dataloader and val_dataloader
    input_preparer.config.train.is_train = True
    df_train = df[folds.kfold != fold]
    train_encodings, train_labels, train_target_labels, train_text_ids = input_preparer.prepare_input(df_train,
                                                                                                      df_train=df_train)
    train_loader = _make_dataloader_from_encodings(train_encodings, train_labels, train_target_labels,
                                                   tokenizer, config,
                                                   is_train=True, shuffle=True, pad_to_batch=pad_to_batch,
                                                   mask_ratio=mask_ratio)
    for x in train_loader:
        print(len(x))
        break

    input_preparer.config.train.is_train = False
    df_val = df[folds.kfold == fold]
    val_encodings, val_labels, val_target_labels, val_text_ids = input_preparer.prepare_input(df_val, df_train=df_train)
    # val_labels = val_labels.tolist()
    # sort val data
    if sort_val_set:
        lengths = [sum(i["attention_mask"]) for i in val_encodings]
        sorted_values = sorted(zip(val_encodings, val_labels, val_target_labels, val_text_ids, lengths),
                               key=lambda x: x[-1])
        val_encodings, val_labels, val_target_labels, val_text_ids = [], [], [], []
        for e, l, tl, t, _ in sorted_values:
            val_encodings.append(e)
            val_labels.append(l)
            val_target_labels.append(tl)
            val_text_ids.append(t)
    val_loader = _make_dataloader_from_encodings(val_encodings, val_labels, val_target_labels, tokenizer, config,
                                                 is_train=True, shuffle=False, pad_to_batch=pad_to_batch,
                                                 mask_ratio=0)
    return (train_loader, train_text_ids), (val_loader, val_text_ids)


if __name__ == '__main__':
    import os

    from transformers import AutoTokenizer
    from omegaconf import OmegaConf

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    config = OmegaConf.load("../../config/deberta_v3_large_reg.yaml")
    df = pd.read_csv("../../input/feedback-prize-english-language-learning/train.csv")
    folds = pd.read_csv("/home/heyao/kaggle/feedback-ells/input/fb3_folds/train_4folds.csv")
    # folds["kfold"] = folds["fold"]
    fold = 0
    tokenizer = AutoTokenizer.from_pretrained(config.model.path)
    config.train.do_mlm = False

    df_train = df[folds.kfold != fold]
    df_val = df[folds.kfold == fold]
    (dataloader, text_ids), (val_loader, val_ids) = make_dataloader(df, df_train, tokenizer, config, folds=folds,
                                                                    fold=fold, sort_val_set=True)
    # dataloader, text_ids = make_dataloader(df, tokenizer, config, folds=folds, fold=fold)
    x, target_y, y = next(iter(dataloader))
    for key in x.keys():
        print(key, x[key].shape)
    print(y.shape, target_y.shape)
    print(tokenizer.decode(x["input_ids"][0]))
    print(tokenizer.decode(x["input_ids"][1]))
    # for x, _ in val_loader:
    #     print(x["attention_mask"].sum(axis=1))
