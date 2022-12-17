"""
@created by: heyao
@created at: 2022-09-05 19:33:47
"""
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

    f = "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv"
    f = f if config.dataset.path == "..." else config.dataset.path
    df = pd.read_csv(f)
    folds = pd.read_csv(config.dataset.fold_file)
    tokenizer_class = AutoTokenizer
    tokenizer = tokenizer_class.from_pretrained(config.model.path, use_fast=True)
    train_fold = config.train.fold_index
    model_class = FeedbackRegressionModule

    train_loader, train_ids = make_dataloader(df, tokenizer, config, folds=None,
                                              fold=None, sort_val_set=True,
                                              pad_to_batch=config.train.get("pad_to_batch", True),
                                              mask_ratio=config.train.get("mask_ratio", 0))
    print(tokenizer.decode(train_loader.dataset[0][0]["input_ids"]))
    config.dataset.n_samples_in_train = len(df)
    print(f"train model with {model_class.__name__}")
    print(f"num samples: {config.dataset.n_samples_in_train}, "
          f"total {len(train_loader)} * {config.train.epochs} = {len(train_loader) * config.train.epochs} steps.")
    # initial some useful variables
    name = f"{config.model.path.split('/')[-1]}_full"
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
        name = f"{config.model.path.split('/')[-1]}_full"
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
    pl.seed_everything(config.train.random_seed)  # to ensure the model weights initial
    model = model_class(config)
    model.backbone.resize_token_embeddings(len(tokenizer))
    if config.train.get("load_from_prev", False):
        prev_path = config.train.prev_path
        print(f"load from {prev_path}")
        state_dict = torch.load(prev_path, map_location="cpu")["state_dict"]
        weights = {k.split(".", 1)[-1]: v for k, v in state_dict.items() if "backbone" in k}
        model.backbone.load_state_dict(weights)
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
        pl.seed_everything(config.train.random_seed)  # to ensure the data order

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
    if config.train.freeze_embeddings:
        model.backbone.embeddings.requires_grad_(False)
    if config.train.freeze_encoders > 0:
        model.backbone.encoder.layer[:config.train.freeze_encoders].requires_grad_(False)
    trainer.fit(model, train_dataloaders=train_loader)
    # save the last checkpoint
    torch.save({"state_dict": model.state_dict(), "metric": model.best_score},
               f"{os.path.join(save_to, f'{name}.ckpt')}")
    print(f"finished use {time.perf_counter() - t0:.1f} seconds.")
