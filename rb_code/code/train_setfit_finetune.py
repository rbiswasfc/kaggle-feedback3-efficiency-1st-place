import gc
import json
import math
import os
import pdb
import random
import time
from itertools import chain

import hydra
import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_cosine_schedule_with_warmup

try:
    from setfit.fb_dataloader import CustomDataCollatorWithPadding
    from setfit.fb_dataset import FeedbackDataset
    from setfit.fb_model import AWP, FeedbackModelSetFit
    from utils.optim_utils import get_optimizer
    from utils.train_utils import (EMA, AverageMeter, apply_mixout, get_lr,
                                   get_score, init_wandb,
                                   print_gpu_utilization, save_checkpoint,
                                   seed_everything)

except Exception as e:
    print(e)
    raise ImportError

#-------- Utils -----------------------------------------------------------------#


def print_line():
    prefix, unit, suffix = "#", "--", "#"
    print(prefix + unit*50 + suffix)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm%ds' % (m, s)
#-------- Main Function ---------------------------------------------------------#


@hydra.main(version_base=None, config_path="../conf/setfit", config_name="conf_stage_2")
def run_training(cfg):
    #------- Runtime Configs -----------------------------------------------------#
    print_line()
    if cfg.use_random_seed:
        seed = random.randint(401, 999)
        cfg.seed = seed

    print(f"setting seed: {cfg.seed}")
    seed_everything(cfg.seed)

    if cfg.all_data:
        print("running training with all data...")
        fold = 0
        cfg.train_folds = [i for i in range(cfg.fold_metadata.n_folds)]
        cfg.valid_folds = [fold]
        cfg.outputs.model_dir = os.path.join(cfg.outputs.model_dir, f"all_data_training/seed_{cfg.seed}")
    else:
        fold = cfg.fold
        cfg.train_folds = [i for i in range(cfg.fold_metadata.n_folds) if i != fold]
        cfg.valid_folds = [fold]

    print(f"train folds: {cfg.train_folds}")
    print(f"valid folds: {cfg.valid_folds}")

    target_names = cfg.model.target_names
    print(f"target column names: {target_names}")
    # ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    #------- folder management --------------------------------------------------#
    # os.makedirs(cfg.outputs.output_dir, exist_ok=True)
    os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    #------- load data ----------------------------------------------------------#
    print_line()
    cfg_dataset = cfg.competition_dataset
    df = pd.read_csv(os.path.join(cfg_dataset.data_dir, cfg_dataset.train_path))
    fold_df = pd.read_parquet(os.path.join(cfg.fold_metadata.fold_dir, cfg.fold_metadata.fold_path))
    df = pd.merge(df, fold_df, on="text_id", how="left")

    if cfg.debug:
        print("DEBUG Mode: sampling 10% examples from train data")
        num_examples = max(int(0.1*len(df)), 1)
        df = df.sample(num_examples)
        print(df.head())

    print("creating the datasets and data loaders...")
    train_df = df[df["kfold"].isin(cfg.train_folds)].copy()
    valid_df = df[df["kfold"].isin(cfg.valid_folds)].copy()

    print(f"shape of train data: {train_df.shape}")
    print(f"shape of valid data: {valid_df.shape}")
    print_line()

    #------- mix pl data ------------------------------------------------------#
    if cfg.ensemble_pl.use_pl:
        pl_dfs = []
        for fpath in cfg.ensemble_pl.pl_paths:
            pl_dfs.append(pd.read_csv(fpath))

        raw_df = pd.read_csv(cfg.ensemble_pl.pl_raw_path)
        mapping = dict(zip(raw_df["text_id"], raw_df["full_text"]))
        pseudo_df = pd.concat(pl_dfs)
        # pdb.set_trace()
        pseudo_df = pseudo_df.groupby('text_id')[["cohesion", "syntax", "vocabulary",
                                                  "phraseology", "grammar", "conventions"]].agg(np.mean).reset_index()
        pseudo_df["full_text"] = pseudo_df["text_id"].map(mapping)

        # remove valid samples + sample
        pseudo_df["text_id"] = pseudo_df["text_id"].apply(lambda x: x.split("_")[0])
        print(f"shape of pseudo df before removing valid ids: {pseudo_df.shape}")
        valid_ids = valid_df["text_id"].unique().tolist()
        pseudo_df = pseudo_df[~pseudo_df["text_id"].isin(valid_ids)].copy()
        print(f"shape of pseudo df after removing valid ids: {pseudo_df.shape}")
        pseudo_df = pseudo_df.sample(cfg.ensemble_pl.num_pl_samples)
        print(f"shape of pseudo df after sampling: {pseudo_df.shape}")
        pseudo_df = pseudo_df.reset_index(drop=True)

        # add pl data with training data
        train_df = pd.concat([train_df, pseudo_df])
        train_df = train_df.reset_index(drop=True)
        print(f"shape of train data after adding pl data: {train_df.shape}")

    #------- dataset ------------------------------------------------------------#

    dataset_creator = FeedbackDataset(cfg)
    train_ds = dataset_creator.get_dataset(train_df, mode="train")
    valid_ds = dataset_creator.get_dataset(valid_df, mode="valid")
    tokenizer = dataset_creator.tokenizer

    # save datasets
    train_ds.save_to_disk(os.path.join(cfg.outputs.model_dir, f"train_dataset_fold_{fold}"))
    valid_ds.save_to_disk(os.path.join(cfg.outputs.model_dir, f"valid_dataset_fold_{fold}"))
    cfg.model.len_tokenizer = len(tokenizer)

    #------- data loaders --------------------------------------------------------#

    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    train_ds.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'labels']
    )

    # sort valid dataset for faster evaluation
    valid_ds = valid_ds.sort("input_length")
    valid_text_ids = valid_ds["text_id"]

    valid_ds.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'labels']
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.train_bs,
        shuffle=True,
        collate_fn=data_collector,
        pin_memory=True,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.valid_bs,
        shuffle=False,
        collate_fn=data_collector,
        pin_memory=True,
    )

    print("data preparation done...")
    print_line()

    #------- Config ---------------------------------------------------------------------#
    print("config for the current run")
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4))

    #------- Model ---------------------------------------------------------------------#

    print_line()
    print("creating the feedback model...")
    model = FeedbackModelSetFit(cfg_dict)
    print_line()

    # if cfg.model.load_from_ckpt:
    #     print("loading model from previously trained checkpoint...")
    #     checkpoint = cfg.model.ckpt_path
    #     ckpt = torch.load(checkpoint)
    #     model.load_state_dict(ckpt['state_dict'])
    #     print(f"At onset model performance on validation set = {ckpt['lb']}")
    #     del ckpt
    #     gc.collect()

    print("loading sentence transformer from previous checkpoint...")
    checkpoint = cfg_dict["model"]["ckpt_path"]
    ckpt = torch.load(checkpoint)
    model.sentence_transformer.load_state_dict(ckpt['state_dict'])
    del ckpt
    gc.collect()

    if cfg.train_params.use_mixout:
        print("=="*40)
        model = apply_mixout(model, p=cfg.train_params.mixout_prob)
        print("training will use mixout as regularization instead of dropout")
        print("=="*40)

    #------- Optimizer -----------------------------------------------------------------#
    print_line()
    print("creating the optimizer...")
    optimizer = get_optimizer(model, cfg_dict)

    #------- Scheduler -----------------------------------------------------------------#
    print_line()
    print("creating the scheduler...")

    num_epochs = cfg_dict["train_params"]["num_epochs"]
    grad_accumulation_steps = cfg_dict["train_params"]["grad_accumulation"]
    warmup_pct = cfg_dict["train_params"]["warmup_pct"]

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch

    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    # #------- AWP --------------------------------------------------------------------------#

    AWP_FLAG = False

    # AWP
    if cfg.awp.use_awp:
        awp = AWP(model, optimizer, adv_lr=cfg.awp.adv_lr, adv_eps=cfg.awp.adv_eps)

    #------- Accelerator ---------------------------------------------------------------------#
    print_line()
    print("accelerator setup...")
    if cfg_dict["train_params"]["use_fp16"]:
        print("using mixed precision training")
        accelerator = Accelerator(fp16=True)  # (fp16=True)
    else:
        accelerator = Accelerator()  # cpu = True

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization()
    print_line()

    #------- Wandb ----------------------------------------------------------------------------#
    if cfg.use_wandb:
        print("initializing wandb run...")
        init_wandb(cfg_dict)

    #------- training setup -------------------------------------------------------------------#
    best_score = 1e6
    save_trigger = cfg_dict["train_params"]["save_trigger"]
    org_eval_freq = cfg_dict["train_params"]["eval_frequency"]

    tracker = 0
    wandb_step = 0

    best_cohesion = 1e6
    best_syntax = 1e6
    best_vocabulary = 1e6
    best_phraseology = 1e6
    best_grammar = 1e6
    best_conventions = 1e6

    #------- EMA -----------------------------------------------------------------------------#
    if cfg.train_params.use_ema:
        print_line()
        decay_rate = 0.99
        # torch.exp(torch.log(cfg.train_params.ema_prev_epoch_weight) / num_update_steps_per_epoch)
        ema = EMA(model, decay=decay_rate)
        ema.register()

        print(f"EMA will be used during evaluation with decay {round(decay_rate, 4)}...")
        print_line()

    #------- Training -----------------------------------------------------------------------#
    start_time = time.time()
    print_line()
    print("freezing sentence transformer backbone...")
    model.freeze(n_layers=-1, freeze_all=True)
    print_line()

    for epoch in range(num_epochs):

        # unfreeze top layers
        print(f"\n========== EPOCH {epoch+1} ============================\n")
        if epoch == 0:  # 1
            print("unfreezing top 1 layers of the backbone...")
            model.unfreeze(n_layers=1)
        # # elif epoch == 1:

        # #     print("unfreezing entire backbone...")
        # #     model.unfreeze(n_layers=-1, unfreeze_all=True)

        #     # print("unfreezing top 12 layers of the backbone...")
        #     # model.unfreeze(n_layers=12)
        # # elif epoch == 2:
        # #     print("unfreezing entire backbone...")
        # #     model.unfreeze(n_layers=-1, unfreeze_all=True)
        # else:
        #     pass
        # print_line()

        # AWP Flag check
        if (cfg.awp.use_awp) & (epoch >= cfg.awp.awp_trigger_epoch):
            print("AWP is triggered...")
            AWP_FLAG = True

        # set evaluation frequency
        if epoch < cfg_dict["train_params"]["full_eval_start_epoch"]:
            cfg_dict["train_params"]["eval_frequency"] = org_eval_freq * 4
        else:
            cfg_dict["train_params"]["eval_frequency"] = org_eval_freq

        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        n_total = num_update_steps_per_epoch
        progress_bar = tqdm(range(n_total))

        loss_meter = AverageMeter()
        cohesion_loss_meter = AverageMeter()
        syntax_loss_meter = AverageMeter()
        vocabulary_loss_meter = AverageMeter()
        phraseology_loss_meter = AverageMeter()
        grammar_loss_meter = AverageMeter()
        conventions_loss_meter = AverageMeter()

        # Training
        model.train()

        for step, batch in enumerate(train_dl):
            logits, loss, loss_dict = model(**batch)
            accelerator.backward(loss)

            if AWP_FLAG:
                awp.attack_backward(batch, accelerator)

            if (step + 1) % grad_accumulation_steps == 0:
                if cfg_dict["train_params"]["use_fp16"]:
                    pass  # not doing grad clipping in mixed precision mode
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        cfg_dict["optimizer"]["grad_clip"],
                    )

                # take optimizer and scheduler steps
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())

                if cfg.train_params.use_ema:
                    ema.update()

                try:
                    cohesion_loss_meter.update(loss_dict["cohesion"].item())
                    syntax_loss_meter.update(loss_dict["syntax"].item())
                    vocabulary_loss_meter.update(loss_dict["vocabulary"].item())
                    phraseology_loss_meter.update(loss_dict["phraseology"].item())
                    grammar_loss_meter.update(loss_dict["grammar"].item())
                    conventions_loss_meter.update(loss_dict["conventions"].item())
                except Exception as e:
                    pass

                progress_bar.set_description(
                    f"STEP: {(step+1)//grad_accumulation_steps:5}/{n_total:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"Loss: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                wandb_step += 1

                if cfg.use_wandb:
                    wandb.log({"train_loss": round(loss_meter.avg, 5)}, step=wandb_step)
                    wandb.log({"cohesion_loss": round(cohesion_loss_meter.avg, 5)}, step=wandb_step)
                    wandb.log({"syntax_loss": round(syntax_loss_meter.avg, 5)}, step=wandb_step)
                    wandb.log({"vocabulary_loss": round(vocabulary_loss_meter.avg, 5)}, step=wandb_step)
                    wandb.log({"phraseology_loss": round(phraseology_loss_meter.avg, 5)}, step=wandb_step)
                    wandb.log({"grammar_loss": round(grammar_loss_meter.avg, 5)}, step=wandb_step)
                    wandb.log({"conventions_loss": round(conventions_loss_meter.avg, 5)}, step=wandb_step)
                    wandb.log({"lr": get_lr(optimizer)}, step=wandb_step)

            # Evaluation
            if (step + 1) % cfg_dict["train_params"]["eval_frequency"] == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print_gpu_utilization()

                # set model in eval mode
                model.eval()

                # apply ema if it is used
                if cfg.train_params.use_ema:
                    ema.apply_shadow()

                all_preds = []
                all_labels = []

                for _, batch in enumerate(valid_dl):
                    with torch.no_grad():
                        batch_labels = batch["labels"]

                        logits, _, _ = model(**batch)
                        if cfg_dict["model"]["loss_fn"] == "bce":
                            preds = model.lb + (model.ub-model.lb) * torch.sigmoid(logits)
                        else:
                            preds = logits

                    all_preds.append(preds)
                    all_labels.append(batch_labels)

                # pdb.set_trace()
                all_preds = [p.to('cpu').detach().numpy().tolist() for p in all_preds]
                all_preds = np.array(list(chain(*all_preds)))

                all_labels = [p.to('cpu').detach().numpy().tolist() for p in all_labels]
                all_labels = np.array(list(chain(*all_labels)))

                # OOF
                preds_df = pd.DataFrame()  # all_preds)
                preds_df["text_id"] = valid_text_ids
                num_targets = len(target_names)

                for i in range(num_targets):
                    preds_df[target_names[i]] = all_labels[:, i]
                    preds_df[f"pred_{target_names[i]}"] = all_preds[:, i]

                # compute loss
                scores_dict = get_score(all_labels, all_preds)
                print(f">>> Epoch {epoch+1} | Step {step+1} | Elapsed time: {as_minutes(time.time()-start_time)}")
                print(f">>> Current LB = {scores_dict['lb']}")

                if cfg.use_wandb:
                    wandb.log({"lb": scores_dict['lb']}, step=wandb_step)
                    wandb.log({"cohesion_lb": scores_dict['cohesion']}, step=wandb_step)
                    wandb.log({"syntax_lb": scores_dict['syntax']}, step=wandb_step)
                    wandb.log({"vocabulary_lb": scores_dict['vocabulary']}, step=wandb_step)
                    wandb.log({"phraseology_lb": scores_dict['phraseology']}, step=wandb_step)
                    wandb.log({"grammar_lb": scores_dict['grammar']}, step=wandb_step)
                    wandb.log({"conventions_lb": scores_dict['conventions']}, step=wandb_step)

                # save model
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': step + 1,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'lb': scores_dict['lb'],
                }

                is_best = False
                if scores_dict['lb'] < best_score:
                    best_score = scores_dict['lb']
                    is_best = True
                    tracker = 0
                else:
                    tracker += 1

                if scores_dict["cohesion"] < best_cohesion:
                    best_cohesion = scores_dict["cohesion"]

                if scores_dict["syntax"] < best_syntax:
                    best_syntax = scores_dict["syntax"]

                if scores_dict["vocabulary"] < best_vocabulary:
                    best_vocabulary = scores_dict["vocabulary"]

                if scores_dict["phraseology"] < best_phraseology:
                    best_phraseology = scores_dict["phraseology"]

                if scores_dict["grammar"] < best_grammar:
                    best_grammar = scores_dict["grammar"]

                if scores_dict["conventions"] < best_conventions:
                    best_conventions = scores_dict["conventions"]

                if is_best:
                    if best_score < save_trigger:
                        save_checkpoint(cfg_dict, model_state, is_best=is_best)
                    preds_df.to_csv(os.path.join(cfg_dict["outputs"]["model_dir"], f"oof_df_fold_{fold}.csv"), index=False)
                else:
                    print(f">>> patience reached {tracker}/{cfg_dict['train_params']['patience']}")
                    print(f">>> current best score: {round(best_score, 4)}")

                if cfg.use_wandb:
                    wandb.log({"best_lb": best_score}, step=wandb_step)
                    if is_best:
                        wandb.log({"cohesion@best_lb": scores_dict['cohesion']}, step=wandb_step)
                        wandb.log({"syntax@best_lb": scores_dict['syntax']}, step=wandb_step)
                        wandb.log({"vocabulary@best_lb": scores_dict['vocabulary']}, step=wandb_step)
                        wandb.log({"phraseology@best_lb": scores_dict['phraseology']}, step=wandb_step)
                        wandb.log({"grammar@best_lb": scores_dict['grammar']}, step=wandb_step)
                        wandb.log({"conventions@best_lb": scores_dict['conventions']}, step=wandb_step)

                    wandb.log({"cohesion_best": best_cohesion}, step=wandb_step)
                    wandb.log({"syntax_best": best_syntax}, step=wandb_step)
                    wandb.log({"vocabulary_best": best_vocabulary}, step=wandb_step)
                    wandb.log({"phraseology_best": best_phraseology}, step=wandb_step)
                    wandb.log({"grammar_best": best_grammar}, step=wandb_step)
                    wandb.log({"conventions_best": best_conventions}, step=wandb_step)

                if (cfg.awp.use_awp) & (best_score <= cfg.awp.awp_trigger):
                    print("AWP is triggered...")
                    AWP_FLAG = True

                torch.cuda.empty_cache()

                if cfg.train_params.use_ema:
                    ema.restore()

                model.train()
                print("GPU Utilization after evaluation...")
                print_gpu_utilization()
                print_line()

                if tracker >= cfg_dict['train_params']['patience']:
                    print("stopping early")
                    model.eval()
                    return


if __name__ == "__main__":
    run_training()
