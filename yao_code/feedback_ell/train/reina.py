"""
@created by: heyao
@created at: 2022-09-05 19:33:47
"""
from collections import Counter

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None

import os
import gc
import time
from argparse import ArgumentParser

import torch
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from feedback_ell.modules.regression.reina.both import ReinaRegressionModule
from feedback_ell.nn import apply_mixout
from feedback_ell.utils import label_columns
from feedback_ell.utils.dir import _get_next_version
from feedback_ell.utils.savor import save_yaml
from feedback_ell.dataloaders.reina import make_dataloader
from feedback_ell.utils.stable_training import init_weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if __name__ == '__main__':
    t0 = time.perf_counter()
    parser = ArgumentParser()
    parser.add_argument("--config", default=None, required=False)
    args, unknown_args = parser.parse_known_args()

    gc.enable()
    if args.config is None:
        config = OmegaConf.load("../../config/deberta_v3_base_reg.yaml")
        print("dont pass config, DEBUG mode enable")
    else:
        config = OmegaConf.load(args.config)
        config.merge_with_dotlist(unknown_args)

    # from tqdm.auto import tqdm
    #
    # for _ in tqdm(range(3600 + 1200)):
    #     time.sleep(1)
    f = "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv"
    df = pd.read_csv(f)
    folds = pd.read_csv(config.dataset.fold_file)
    if "coco" in config.model.path:
        tokenizer_class = COCOLMTokenizer
    else:
        tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(config.model.path, use_fast=True)
    train_fold = config.train.fold_index
    pools = ["mean", "max", "cls"]
    if config.model.get("head", "") == "sep_attention" and any(i in config.model.pooling for i in pools):
        raise ValueError(f"sep_attention must use a pooler out of {', '.join(pools)}")

    for fold in range(config.train.num_folds):
        if fold not in train_fold:
            continue
        model_class = ReinaRegressionModule

        if config.train.get("with_description", False):
            s = [
                # "Command of grammar and usage with few or no errors.",
                # "Minimal errors in grammar and usage.",
                # "Some errors in grammar and usage.",
                # "Many errors in grammar and usage.",
                # "Errors in grammar and usage throughout."
                f"How is the {' '.join(label_columns)} of the text?"
            ]
            df["full_text"] = f"{tokenizer.sep_token}".join(s) + tokenizer.sep_token + df["full_text"].str.strip()

        df_train = df[folds.kfold != fold]
        (train_loader, train_ids), (val_loader, val_ids) = make_dataloader(df, df_train, tokenizer, config, folds=folds,
                                                                           fold=fold, sort_val_set=True,
                                                                           pad_to_batch=config.train.get("pad_to_batch", True),
                                                                           mask_ratio=config.train.get("mask_ratio", 0))
        print(tokenizer.decode(train_loader.dataset[0][0]["input_ids"]))
        print("target label:", train_loader.dataset[0][1])
        print("labels:", train_loader.dataset[0][2])
        if config.train.pl.enable and config.train.pl.stage % 2 == 1:
            df = pd.read_csv(config.train.pl.path.format(fold=fold))
            # df = df.sort_values("syntax")
            # sampled = resample(df.syntax, bins=4)
            # df["sampled"] = sampled
            # df = df[df["sampled"] == 1].reset_index(drop=True)
            print(f"train with pl({df.shape})")
            train_loader, train_ids = make_dataloader(df, tokenizer, config)
        config.dataset.n_samples_in_train = int((folds.kfold != fold).sum())
        print(f"train model with {model_class.__name__}")
        print(f"num samples: {config.dataset.n_samples_in_train}, "
              f"total {len(train_loader)} * {config.train.epochs} = {len(train_loader) * config.train.epochs} steps.")
        # initial some useful variables
        name = f"{config.model.path.split('/')[-1]}_fold{fold}"
        if config.train.report_to.tensorboard.enable:
            path = config.train.report_to.tensorboard.path
            version = f"version_{_get_next_version(path)}_{name}"
            logger = TensorBoardLogger(save_dir=path, version=version)
        elif config.train.report_to.wandb.enable:
            logger = WandbLogger(project="feedback3", name=name)
        else:
            raise ValueError(f"report default to tensorboard")
        print(f"train with config: {config}")
        save_to = config.train.get("save_to", "./weights")
        os.makedirs(os.path.abspath(save_to), exist_ok=True)
        print(f"model save to {os.path.abspath(save_to)}")
        if config.train.pl.enable:
            name = f"{config.model.path.split('/')[-1]}_{model_type}_fold{fold}"
        if config.train.trained_target and len(config.train.trained_target) == 1:
            name = f"{config.model.path.split('/')[-1]}_{model_type}_target{config.train.trained_target[0]}_fold{fold}"
        checkpoint = ModelCheckpoint(
            dirpath=save_to,
            filename=name,
            monitor="val/score",
            mode="min",
            save_weights_only=True,
            save_last=None,
            save_on_train_epoch_end=None,
        )
        callbacks = [
            checkpoint,
            LearningRateMonitor("step"),
            # StochasticWeightAveraging(swa_epoch_start=4, annealing_epochs=4, device="cpu")
        ]
        max_grad_norm = config.train.max_grad_norm
        if config.train.ema.enable:
            max_grad_norm = None
        trainer = pl.Trainer(
            max_epochs=config.train.epochs, accelerator=config.train.device, devices=1,
            val_check_interval=config.train.validation_interval,
            accumulate_grad_batches=config.train.accumulate_grad,
            precision=config.train.precision,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=max_grad_norm,
            enable_progress_bar=config.train.get("enable_progress_bar", False),
            num_sanity_val_steps=2 if config.train.get("debug", False) else 0,
        )
        # start training
        pl.seed_everything(config.train.random_seed + fold)  # to ensure the model weights initial
        model = model_class(config)
        model.backbone.resize_token_embeddings(len(tokenizer))
        if config.train.get("load_from_prev", False):
            prev_path = config.train.prev_path
            print(f"load from {prev_path}")
            state_dict = torch.load(prev_path, map_location="cpu")["state_dict"]
            # weights = {k.split(".", 2)[-1]: v for k, v in state_dict.items() if "deberta" in k}
            weights = {k.split(".", 1)[-1]: v for k, v in state_dict.items() if "backbone" in k}
            # 50265
            # weights["embeddings.word_embeddings.weight"] = weights["embeddings.word_embeddings.weight"][:128001, :]
            model.backbone.load_state_dict(weights)
        if config.train.pl.enable and config.train.pl.stage % 2 == 0:
            model.load_state_dict(torch.load(config.train.pl.prev_model_path.format(fold=fold),
                                             map_location="cpu")["state_dict"])
            config.train.reinit_weights = False
        if config.train.trained_target:
            print(f"train model with label: {[label_columns[i] for i in config.train.trained_target]}")
        if config.train.get("save_config", False):
            save_yaml(config, os.path.join(save_to, args.config.split("/")[-1]))
        # if "roberta-large" in config.model.path and config.reinit_from_sb:
        #     print("load state dict from sentence transformer")
        #     from sentence_transformers import SentenceTransformer
        #
        #     model_path = "/media/heyao/42f00068-ba8d-48dd-8719-61eda5244b8d/pretrained-models/roberta-large-nli-stsb-mean-tokens"
        #     smodel = SentenceTransformer(model_path)
        #     state_dict = {k: v for k, v in smodel[0].auto_model.state_dict().items() if "pooler" not in k}
        #     model.backbone.load_state_dict(state_dict)

        if config.train.reinit_weights:
            initializer_range = config.train.get("initializer_range", 0.02)
            init_weights(model.head, initializer_range=initializer_range)
            init_weights(model.customer_pooling, initializer_range=initializer_range)
        if config.train.mixout.enable:
            apply_mixout(model.backbone, p=config.train.mixout.p)
        if config.train.ensure_data_order:
            print(f"ensure data order of training, will set seed again.")
            pl.seed_everything(config.train.random_seed + fold)  # to ensure the data order

        if bnb is not None:
            if hasattr(model.backbone, "embeddings"):
                embs = model.backbone.embeddings
            elif hasattr(model.backbone, "shared"):
                embs = model.backbone.shared
            for emb_type in ["word", "position", "token_type"]:
                attr_name = f"{emb_type}_embeddings"

                # Note: your model type might have a different path to the embeddings
                if hasattr(embs, attr_name) and getattr(embs, attr_name) is not None:
                    bnb.optim.GlobalOptimManager.get_instance().register_module_override(
                        getattr(embs, attr_name), 'weight', {'optim_bits': 32}
                    )
            del embs
            gc.collect()
        # freeze
        if "bart" in config.model.path:
            if config.train.freeze_embeddings:
                model.backbone.shared.requires_grad_(False)
                model.backbone.encoder.embed_tokens.requires_grad_(False)
                model.backbone.encoder.embed_positions.requires_grad_(False)
            if config.train.freeze_encoders > 0:
                model.backbone.encoder.layers[:config.train.freeze_encoders].requires_grad_(False)
        else:
            if config.train.freeze_embeddings:
                model.backbone.embeddings.requires_grad_(False)
            if config.train.freeze_encoders > 0:
                model.backbone.encoder.layer[:config.train.freeze_encoders].requires_grad_(False)
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        # save the last checkpoint
        if config.train.pl.enable and config.train.pl.stage % 2 == 1:
            torch.save({"state_dict": model.state_dict(), "metric": model.best_score}, f"{os.path.join(save_to, f'{name}.ckpt')}")
    print(f"finished use {time.perf_counter() - t0:.1f} seconds.")
