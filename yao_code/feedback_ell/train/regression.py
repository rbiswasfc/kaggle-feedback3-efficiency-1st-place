"""
@created by: heyao
@created at: 2022-09-05 19:33:47

This is the main file. The following items are defined in the different modules.
- model class: `feedback_ell.modules`: the final solution use `FeedbackRegressionModule` only.
- inputs: `feedback_ell.dataloaders.regression.make_dataloader`
- optimizer, scheduler, training steps, ... are defined in the `feedback_ell.modules.base.BaseLightningModule`
"""
from feedback_ell.modules.regression.simple import FeedbackRegressionModuleWithPLWeight

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

from feedback_ell.modules import FeedbackRegressionModule
from feedback_ell.nn import apply_mixout
from feedback_ell.utils import label_columns
from feedback_ell.utils.dir import _get_next_version
from feedback_ell.utils.savor import save_yaml
from feedback_ell.dataloaders.regression import make_dataloader
from feedback_ell.utils.stable_training import init_weights

os.environ["TOKENIZERS_PARALLELISM"] = "false"
if __name__ == '__main__':
    t0 = time.perf_counter()
    ###################################################################
    # =========== get argument from config and command line ========= #
    ###################################################################
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

    ###################################################################
    # ================ prepare raw data and tokenizer =============== #
    ###################################################################
    f = "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv"
    f = f if config.dataset.path.strip() == "..." else config.dataset.path
    df = pd.read_csv(f)
    folds = pd.read_csv(config.dataset.fold_file)
    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(config.model.path, use_fast=True)
    train_fold = config.train.fold_index

    # fold train
    for fold in range(config.train.num_folds):
        if fold not in train_fold:
            continue
        # define model class with different settings, only use `FeedbackRegressionModule` in the final solution
        model_class = FeedbackRegressionModule
        model_type = config.model.get("type", "regression")
        (train_loader, train_ids), (val_loader, val_ids) = make_dataloader(df, tokenizer, config, folds=folds,
                                                                           fold=fold, sort_val_set=True,
                                                                           pad_to_batch=config.train.get("pad_to_batch", True),
                                                                           mask_ratio=config.train.get("mask_ratio", 0))
        print(tokenizer.decode(train_loader.dataset[0][0]["input_ids"]))
        ###################################################################
        # ======================== train with PL ======================== #
        ###################################################################
        if config.train.pl.enable and config.train.pl.stage % 2 == 1:
            # default to min dict PL
            if config.train.pl.get("min_dist", True):
                import json

                df_pl = pd.read_csv(config.train.pl.path.format(fold=fold))
                if config.train.pl.get("groupall", False):
                    ids = []
                    for i in range(4):
                        with open(config.train.pl.json_path.format(fold=i), "r") as f:
                            ids.append(json.load(f))
                    ids = set(ids[0] + ids[1] + ids[2] + ids[3])
                    print(f"total ids: {len(ids)}")
                else:
                    with open(config.train.pl.json_path.format(fold=fold), "r") as f:
                        ids = json.load(f)
                df_pl = df_pl[df_pl.text_id.isin(ids)].reset_index(drop=True)
                df_pl = df_pl.merge(pd.read_csv(config.train.pl.text_path), how="left", on="text_id")
                # df = df.sort_values("syntax")
                # sampled = resample(df.syntax, bins=4)
                # df["sampled"] = sampled
                # df = df[df["sampled"] == 1].reset_index(drop=True)
                print(f"train with pl({df_pl.shape})")
                df_train = df[folds.kfold != fold].reset_index(drop=True)
                print(df_pl.head(1).T)
                df_pl["weight"] = config.train.pl.weight
                df_train["weight"] = 1.0
                # df = pd.concat([df_pl, df_train], axis=0).reset_index(drop=True)
                df = df_pl.reset_index(drop=True)
                train_loader, train_ids = make_dataloader(df, tokenizer, config)
                model_class = FeedbackRegressionModuleWithPLWeight
            else:
                df_pl = pd.read_csv(config.train.pl.path.format(fold=fold))
                train_loader, train_ids = make_dataloader(df_pl, tokenizer, config)
        config.dataset.n_samples_in_train = int((folds.kfold != fold).sum())
        print(f"train model with {model_class.__name__}")
        print(f"num samples: {config.dataset.n_samples_in_train}, "
              f"total {len(train_loader)} * {config.train.epochs} = {len(train_loader) * config.train.epochs} steps.")
        # initial some useful variables, e.g. model save name, logger, ...
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
        # TODO: 预测伪标签，oof的配置
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
        # train with PL (if stage2)
        if config.train.pl.enable and config.train.pl.stage % 2 == 0:
            model.load_state_dict(torch.load(config.train.pl.prev_model_path.format(fold=fold),
                                             map_location="cpu")["state_dict"])
            config.train.reinit_weights = False
        if config.train.trained_target:
            print(f"train model with label: {[label_columns[i] for i in config.train.trained_target]}")
        if config.train.get("save_config", False):
            save_yaml(config, os.path.join(save_to, args.config.split("/")[-1]))

        if config.train.reinit_weights:
            initializer_range = config.train.get("initializer_range", 0.02)
            init_weights(model.head, initializer_range=initializer_range)
            init_weights(model.customer_pooling, initializer_range=initializer_range)
        if config.train.mixout.enable:
            apply_mixout(model.backbone, p=config.train.mixout.p)
        if config.train.ensure_data_order:
            print(f"ensure data order of training, will set seed again.")
            pl.seed_everything(config.train.random_seed + fold)  # to ensure the data order

        # if apply bitsandbytes, need to update `optim_bits` if use mix-precision
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
        if config.train.pl.enable and config.train.pl.stage % 2 == 1 and config.train.pl.save_last_checkpoint:
            torch.save({"state_dict": model.state_dict(), "metric": model.best_score},
                       f"{os.path.join(save_to, f'{name}.ckpt')}")
    print(f"finished use {time.perf_counter() - t0:.1f} seconds.")
