"""
@created by: heyao
@created at: 2022-09-07 16:03:41
"""
import os
from argparse import ArgumentParser
import gc
import warnings

from omegaconf import OmegaConf
import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from feedback_ell.input_preparer.external_tags import ExternalTagRegressionInputPreparer
from feedback_ell.utils.metrics import competition_score
from feedback_ell.utils.dataset.simple import CompetitionDataset
from feedback_ell.modules import FeedbackRegressionModule
from feedback_ell.input_preparer import RegressionInputPreparer
from feedback_ell.utils.collators.sequence_bucket import SequenceBucketPadCollator, MaxSeqLenPadCollator

warnings.filterwarnings("ignore")

print("torch version:", torch.__version__)
print("cudnn version:", torch.backends.cudnn.version())
device = "cuda"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def infer_fn(model, dataloader, config):
    model.eval()
    predictions = []
    with torch.no_grad():
        for x, _ in tqdm(dataloader, total=len(dataloader)):
            for k, v in x.items():
                x[k] = v.to(device)
            y_preds = model([x])
            predictions.append(y_preds.cpu().numpy())
    return np.concatenate(predictions)


def _make_dataloader_from_encodings(encodings, tokenizer, config, is_train, shuffle, pad_to_batch=True):
    batch_size = config.train.batch_size
    # num_workers = config.train.num_workers
    num_workers = 2
    collate_class = SequenceBucketPadCollator if pad_to_batch else MaxSeqLenPadCollator
    collate_fn = collate_class(config.train.max_length + 256, tokenizer, is_train=is_train, target_is_float=True)
    dataset = CompetitionDataset(encodings, None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                             collate_fn=collate_fn)
    return dataloader


def make_dataloader(df, tokenizer, config, folds=None, fold=None, pad_to_batch=True):
    if fold is not None and folds is None:
        # no fold info to get train, eval set, raise error
        raise ValueError("folds and fold can't both be None")
    if config.train.get("tag_task", {}).get("enable", False):
        input_preparer = ExternalTagRegressionInputPreparer(tokenizer, config)
    else:
        input_preparer = RegressionInputPreparer(tokenizer, config)
    if fold is None:
        # return full dataloader
        encodings, _, text_ids = input_preparer.prepare_input(df)
        lengths = [sum(i["attention_mask"]) for i in encodings]
        sorted_values = sorted(zip(encodings, text_ids, lengths), key=lambda x: x[-1])
        encodings, text_ids = [], []
        for e, t, _ in sorted_values:
            encodings.append(e)
            text_ids.append(t)

        dataloader = _make_dataloader_from_encodings(encodings, tokenizer, config,
                                                     is_train=False, shuffle=False, pad_to_batch=pad_to_batch)
        return dataloader, text_ids


def sort_prediction(prediction, text_ids, sub, columns):
    df_prediction = sub.copy()
    df_prediction.text_id = text_ids
    if config.get("get_feature", False):
        df_prediction.loc[:, columns] = prediction[:, ]
    else:
        df_prediction.loc[:, columns] = prediction[:, :6]
    df_prediction = pd.merge(sub.copy(), df_prediction[["text_id"] + columns], how="left", on="text_id")
    test_predict = df_prediction[columns].values
    return test_predict


if __name__ == "__main__":
    from glob import glob

    parser = ArgumentParser()
    parser.add_argument("config", nargs="?", default=None)
    parser.add_argument("--model_path", nargs="?", default=None)
    args, unknown_args = parser.parse_known_args()
    if args.model_path is not None:
        # config_name = glob(f"{args.model_path}/*.yaml")[0]
        config = OmegaConf.load(args.config)
        config.merge_with_dotlist(unknown_args)
        config.weights = list(sorted(glob(f"{args.model_path}/*.ckpt")))
        print(config.weights)
    else:
        config = OmegaConf.load(args.config)
        config.merge_with_dotlist(unknown_args)

    print(f"generate prediction for config: {config}")
    if config.debug:
        data = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv").head(99)
    else:
        data = pd.read_csv("../input/feedback-prize-english-language-learning/train.csv")
    folds = pd.read_csv(config.dataset.fold_file)
    if "fold" not in folds.columns:
        folds["fold"] = folds["kfold"]
    tokenizer = AutoTokenizer.from_pretrained(config.model.path, use_fast=True)
    print("start prepare model")
    gc.collect()
    model_class = FeedbackRegressionModule

    model = model_class(config)
    model.backbone.resize_token_embeddings(len(tokenizer))
    gc.collect()
    print("prepare move model to cuda")
    model.to(device)
    gc.collect()
    print("finished initial model")
    sub = data[["text_id"]]
    num_columns = 6
    columns = [f"A{i}" for i in range(num_columns)]
    label_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    n_folds = min([config.train.num_folds, len(config.weights)])
    if config.debug:
        n_folds = 2
    print(f"folds: {n_folds}")
    scores = []

    for i in range(n_folds):
        state_dict = torch.load(config.weights[i], map_location="cpu")["state_dict"]
        print(f"load: {config.weights[i]}")
        model.load_state_dict(state_dict)

        val_idx = folds[folds.fold == i].index.tolist()
        df_val = data.loc[val_idx].reset_index(drop=True)
        dataloader, text_ids = make_dataloader(df_val, tokenizer, config, folds=None, fold=None,
                                               pad_to_batch=config.get("pad_to_batch", True))
        predictions = infer_fn(model, dataloader, config)
        val_sub = sub.loc[val_idx, ["text_id"]].reset_index(drop=True).copy()
        predictions = sort_prediction(predictions, text_ids, val_sub, [f"A{i}" for i in range(6)])
        test_predict = predictions.copy()

        sub.loc[val_idx, label_columns] = test_predict
        # must have score to validation
        score = competition_score(data.loc[val_idx, label_columns].values, predictions)
        scores.append(score)
        print(f"fold {i} score {np.round(score, 4):.4f}")

    sub.to_csv(config.filename, index=False)
    print(f"mean: {np.mean(scores):.4f}, std: {np.std(scores):.4f}")
