"""
@created by: heyao
@created at: 2022-09-08 17:14:45
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

from feedback_ell.utils.metrics import competition_score
from feedback_ell.utils.dataset.simple import CompetitionDataset
from feedback_ell.modules import FeedbackRegressionModule
from feedback_ell.input_preparer import RegressionInputPreparer
from feedback_ell.utils.collators.sequence_bucket import SequenceBucketPadCollator, MaxSeqLenPadCollator
from inference.utils import load_unlabeled_data

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
    num_workers = config.train.num_workers
    collate_class = SequenceBucketPadCollator if pad_to_batch else MaxSeqLenPadCollator
    collate_fn = collate_class(config.train.max_length, tokenizer, is_train=is_train, target_is_float=True)
    dataset = CompetitionDataset(encodings, None)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle,
                                             collate_fn=collate_fn)
    return dataloader


def make_dataloader(df, tokenizer, config, folds=None, fold=None, pad_to_batch=True):
    if fold is not None and folds is None:
        # no fold info to get train, eval set, raise error
        raise ValueError("folds and fold can't both be None")
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
    data = load_unlabeled_data(fb1_path=config.fb1_path, fb3_path=config.fb3_path)
    if config.debug:
        data = data.head(100)

    tokenizer = AutoTokenizer.from_pretrained(config.model.path, use_fast=True)
    print("start prepare model")
    gc.collect()
    model = FeedbackRegressionModule(config)
    model.backbone.resize_token_embeddings(len(tokenizer))
    gc.collect()
    print("prepare move model to cuda")
    model.to(device)
    gc.collect()
    print("finished initial model")
    predictions = []

    sub = data[["text_id"]]
    scores = []
    columns = ["A", "B", "C", "D", "E", "F"]
    dataloader, text_ids = make_dataloader(data, tokenizer, config, folds=None, fold=None, pad_to_batch=True)
    n_folds = min([config.train.num_folds, len(config.weights)])
    if config.debug:
        n_folds = 2

    for i in range(n_folds):
        if ".pth" in config.weights[i]:
            model.load_state_dict(torch.load(config.weights[i], map_location="cpu"))
        else:
            model.load_state_dict(torch.load(config.weights[i], map_location="cpu")["state_dict"])
        prediction = infer_fn(model, dataloader, config)

        df_prediction = data.copy()
        df_prediction.text_id = text_ids
        df_prediction.loc[:, columns] = prediction
        df_prediction = pd.merge(data, df_prediction[["text_id"] + columns], how="left", on="text_id")
        test_predict = df_prediction[columns].values
        sub = data[["text_id", "full_text"]]
        sub.loc[:, ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]] = test_predict

        sub.to_csv(config.filename.format(fold=i), index=False)
