import gc
import os
import pdb
import random
from collections import OrderedDict
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import wandb
import json

from accelerate import Accelerator
from torch.optim.swa_utils import SWALR, AveragedModel
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer

try:
    from src.base_fts.feedback_dataloader import (
        CustomDataCollatorWithPadding, CustomDataCollatorWithPaddingMaskAug)
    from src.base_fts.feedback_dataset import get_dataset
    from src.base_fts.feedback_model import (AWP, FeedbackModel, EMA)
    from src.utils import (AverageMeter, apply_mixout, get_lr,
                               get_scheduler, save_checkpoint, mcrmse_loss_fn, MCRMSE)
    from train_utils import (get_score, init_wandb, parse_args,
                             print_gpu_utilization, read_config,
                             seed_everything)
    from utils.optim_utils import get_optimizer

except Exception:
    raise ImportError


from loguru import logger

CUDA_VISIBLE_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']


def compute_prefix(text, bank):
    text = text.lower()
    text_words = set(text.split(" "))

    unordered_prefix = ""

    for candidate in bank:
        if candidate in text:
            if candidate in unordered_prefix:
                continue
            elif len(candidate.split(" ")) == 1:  # one word
                if candidate not in text_words:
                    continue
            else:
                unordered_prefix += f"[=SEP=] {candidate}"

    segements = unordered_prefix.split("[=SEP=]")
    segements = [s.strip() for s in segements]

    ordering = [(text.index(s), s) for s in segements]
    ordering = sorted(ordering, key=lambda x: x[0])
    ordered_segements = [o[1] for o in ordering]
    to_return = ", ".join(ordered_segements)
    to_return = to_return[1:].strip()  # remove leading comma

    return to_return

def run_training():
    #-------- seed ------------#
    print("=="*40)
    random_seed = random.randint(401, 999)

    args = parse_args()
    config = read_config(args)

    if config["seed"]:
        seed = config["seed"]
    else:
        seed = random_seed

    print(f"setting seed: {seed}")
    seed_everything(seed)

    config["seed"] = seed

    fold = args.fold
    if config["train_full"] :
        config["train_folds"] = [i for i in range(config["n_folds"])]
    else:
        config["train_folds"] = [i for i in range(config["n_folds"]) if i != fold]

    if config["use_aug_data"]:
        config["train_folds"].append(999)
    config["valid_folds"] = [fold]
    config["fold"] = fold
    print(f"train folds: {config['train_folds']}")
    print(f"valid folds: {config['valid_folds']}")
    print("=="*40)

    os.makedirs(config["model_dir"], exist_ok=True)

    # load train data
    df = pd.read_csv(config["fold_path"])
    df["uid"] = df["text_id"]

    if config["debug"]:
        print("DEBUG Mode: sampling examples from train data")
        df = df.sample(min(100, len(df)))

    if config["use_aug_data"]:
        aug_df = pd.read_csv(config["aug_train_path"])
        aug_base_df = pd.read_csv(config["aug_train_base"])
        aug_df = aug_df.merge(aug_base_df, how='left')
        aug_df['original_essay_id'] = aug_df['text_id'].apply(lambda x: x.split('_')[0])

        train_df_pre = df[df["kfold"].isin(config["train_folds"])].copy().reset_index(drop=True)
        train_essay_ids = train_df_pre["text_id"].unique().tolist()
        print(f"Essay shape before train removal: {aug_df.shape}")
        aug_df = aug_df[~aug_df["text_id"].isin(train_essay_ids)].copy()
        print(f"Essay shape after train removal: {aug_df.shape}")

        valid_df_pre = df[df["kfold"].isin(config["valid_folds"])].copy().reset_index(drop=True)
        valid_essay_ids = valid_df_pre["text_id"].unique().tolist()
        print(f"Essay shape before valid removal: {aug_df.shape}")
        aug_df = aug_df[~aug_df["text_id"].isin(valid_essay_ids)].copy()
        print(f"Essay shape after valid removal: {aug_df.shape}")

        """
        # keep 1:1 ratio for same essay id - drop duplicates
        print(f'Essay shape before removing dup train ids: {aug_df.shape}')
        # Shuffle
        aug_df = aug_df.sample(frac=1).reset_index(drop=True)
        # Drop dups
        aug_df.drop_duplicates(subset=['original_essay_id'], inplace=True)
        print(f'Essay shape after removing dup train ids: {aug_df.shape}')
        """
        if config["aug_dist_sample"]:
            with open(config["aug_dist_sample_path"], "r") as f:
                aug_sample_ids = json.load(f)
            aug_df = aug_df[aug_df["text_id"].isin(aug_sample_ids)].copy()
        else:
            # now sample essays
            aug_df = aug_df.sample(min(config["num_samples"], aug_df.shape[0]))
            
        print(f'---' * 5)
        print(f'Augmented {aug_df.shape[0]} essays')
        print(f'---' * 5)

        aug_df = aug_df.reset_index(drop=True)
        aug_df["kfold"] = 999

        df = pd.concat([df, aug_df]).reset_index(drop=True)


    if config["add_topic"]:
        topics_df = pd.read_csv('topics.csv')
        df = df.merge(topics_df, how='left')
        print(df.head())

    if config['add_prefix']:
        with open("../datasets/kw-dataset/kw_fb_cohesion.json", "r") as f:
            kw_cohesion = json.load(f)

        with open("../datasets/kw-dataset/kw_fb_syntax.json", "r") as f:
            kw_syntax = json.load(f)

        with open("../datasets/kw-dataset/kw_fb_vocabulary.json", "r") as f:
            kw_vocabulary = json.load(f)

        with open("../datasets/kw-dataset/kw_fb_phraseology.json", "r") as f:
            kw_phraseology = json.load(f)

        with open("../datasets/kw-dataset/kw_fb_grammar.json", "r") as f:
            kw_grammar = json.load(f)

        with open("../datasets/kw-dataset/kw_fb_conventions.json", "r") as f:
            kw_conventions = json.load(f)

        kw_list = [
            kw_cohesion,
            kw_syntax,
            kw_vocabulary,
            kw_phraseology,
            kw_grammar,
            kw_conventions,
        ]

        fb3_kws = set()
        for ks in kw_list:
            fb3_kws.update(ks)

        fb3_kws = list(fb3_kws)

        filtered_fb3_kws = []
        for kw in fb3_kws:
            if (len(kw) >= 3):  # & (kw not in STOPS):
                filtered_fb3_kws.append(kw)

        bank = sorted(filtered_fb3_kws, key=lambda x: len(x), reverse=True)
        df["prefix"] = df["full_text"].apply(lambda x: compute_prefix(x, bank))

    # create the dataset
    print("creating the datasets and data loaders...")
    train_df = df[df["kfold"].isin(config["train_folds"])].copy()
    valid_df = df[df["kfold"].isin(config["valid_folds"])].copy()


    print(f"shape of train data: {train_df.shape}")
    print(f"shape of valid data: {valid_df.shape}")

    config["target_cols"] = TARGET_COLUMNS

    train_ds_dict = get_dataset(config, train_df, mode="train")
    valid_ds_dict = get_dataset(config, valid_df, mode="valid")

    tokenizer = train_ds_dict["tokenizer"]
    train_dataset = train_ds_dict["dataset"]
    valid_dataset = valid_ds_dict["dataset"]

    config["len_tokenizer"] = len(tokenizer)
    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'features']
    )

    # sort valid dataset for faster evaluation
    valid_dataset = valid_dataset.sort("input_length")
    valid_text_ids = valid_dataset["text_id"]

    valid_dataset.set_format(
        type=None,
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels', 'features']
    )

    if config["use_mask_aug"]:
        data_collector_train = CustomDataCollatorWithPaddingMaskAug(tokenizer=tokenizer)
    else:
        data_collector_train = CustomDataCollatorWithPadding(tokenizer=tokenizer)

    config["len_tokenizer"] = len(tokenizer)

    train_dl = DataLoader(
        train_dataset,
        batch_size=config["train_bs"],
        shuffle=True,
        collate_fn=data_collector_train,
        pin_memory=True,
    )
    valid_dl = DataLoader(
        valid_dataset,
        batch_size=config["valid_bs"],
        shuffle=False,
        collate_fn=data_collector,
        pin_memory=True,
    )
    print("data preparation done...")
    print("=="*40)

    # create the model and optimizer
    print("creating the model, optimizer and scheduler...")
    model = FeedbackModel(config)

    #if config["debug"]:
    print(model)

    if config['eval_frequency']:
        eval_steps = int(len(train_dl) / config['eval_frequency'])
    else:
        eval_steps = 100000000

    logger.debug(f'Eval steps: {eval_steps}')

    if config["use_mixout"]:
        print("=="*40)
        model = apply_mixout(model, p=config["mixout_prob"])
        print("training will use mixout as regularization instead of dropout")
        print("=="*40)



    optimizer = get_optimizer(model, config)

    swa_scheduler = SWALR(optimizer, swa_lr=config["swa_lr"], anneal_epochs=config["swa_anneal_epochs"])
    swa_model = AveragedModel(model)

    # prepare the training
    num_epochs = config["num_epochs"]
    grad_accumulation_steps = config["grad_accumulation"]
    warmup_pct = config["warmup_pct"]

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)

    AWP_FLAG = False
    SWA_FLAG = False

    # AWP
    if config["use_awp"]:
        awp = AWP(model, optimizer, adv_lr=config["adv_lr"], adv_eps=config["adv_eps"])
        assert config["grad_accumulation"] == 1, "Grad accumulation not supported with AWP"

    # accelerator
    if config["use_fp16"]:
        print("using mixed precision training")
        accelerator = Accelerator(fp16=True)  # (fp16=True)
    else:
        accelerator = Accelerator()  # (fp16=True)

    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    if config["use_ema"]:
        ema = EMA(model, config["ema_decay_rate"])
        ema.register()

    print("model preparation done...")
    print(f"current GPU utilization...")
    print_gpu_utilization(CUDA_VISIBLE_DEVICES)
    print("=="*40)

    # wandb
    if args.use_wandb:
        print("initializing wandb run...")
        init_wandb(config)
        # Save all files that currently exist containing the substring "ckpt"
        wandb.save(f"{config['code_dir']}/*")
        wandb.save(f"{config['code_dir']}/src/*")
        wandb.save(f"{config['code_dir']}/src/fpe_fast/*")

        wandb.save(f"{config['code_dir']}/configs/*")
        wandb.watch(model, log_freq=10)

    # training
    best_score = 1e6
    scores = {}
    best_scores = {}
    ell_score_columns = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    for ell_score in ell_score_columns:
        best_scores[ell_score] = 1e6
        scores[ell_score] = 1e6

    save_trigger = config["save_trigger"]
    tracker = 0
    wandb_step = 0

    for epoch in range(num_epochs):
        """
        if (config["use_awp"]) & (epoch >= config["awp_trigger_epoch"]):
            print("AWP is triggered...")
            AWP_FLAG = True
        """

        if epoch >= config["swa_trigger_epoch"]:
            print("SWA is triggered...")
            SWA_FLAG = True

        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()
        val_loss_meter = AverageMeter()

        # Training
        model.train()

        for step, batch in enumerate(train_dl):
            logits, loss, _ = model(**batch)
            # pdb.set_trace()

            if args.use_wandb:
                wandb.log({"Loss": loss.item()}, step=wandb_step)

            accelerator.backward(loss)


            if AWP_FLAG:
                awp.attack_backward(batch, accelerator)

            if (step + 1) % grad_accumulation_steps == 0:
                if config["use_fp16"]:
                    pass  # not doing grad clipping in mixed precision mode
                else:
                    if config["grad_clip"]:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            config["grad_clip"],
                        )

                # take optimizer and scheduler steps
                optimizer.step()

                if not SWA_FLAG:
                    scheduler.step()
                else:
                    swa_scheduler.step()

                optimizer.zero_grad()

                if config["use_ema"]:
                    ema.update()

                loss_meter.update(loss.item())


                progress_bar.set_description(
                    f"STEP: {(step+1)//grad_accumulation_steps:5}/{num_update_steps_per_epoch:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"TL: {loss_meter.avg:.4f}. "
                )

                progress_bar.update(1)
                wandb_step += 1

                if args.use_wandb:
                    wandb.log({"Train Loss": loss_meter.avg}, step=wandb_step)
                    wandb.log({"LR": get_lr(optimizer)}, step=wandb_step)


            if ((step + 1) % eval_steps) == 0 or ((len(train_dl)-1) - step) == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print("\n")
                print_gpu_utilization(CUDA_VISIBLE_DEVICES)

                model.eval()

                if config["use_ema"]:
                    ema.apply_shadow()

                all_preds = []
                all_labels = []
                prediction = []

                for _, batch in enumerate(valid_dl):
                    with torch.no_grad():
                        logits, loss, _ = model(**batch)
                        val_loss_meter.update(loss.item())
                        batch_preds = logits

                        for pred in batch_preds.cpu():
                            prediction.append(np.array([i for i in np.array(pred)]))
                        batch_labels = batch['labels']

                    #all_preds.append(batch_preds)
                    all_labels.append(batch_labels)


                all_preds = np.array(prediction)

                all_labels = [p.to('cpu').detach().numpy().tolist() for p in all_labels]
                all_labels = list(chain(*all_labels))

                preds_df = pd.DataFrame()
                preds_df['text_id'] = valid_text_ids
                num_targets = len(TARGET_COLUMNS)
                for i in range(num_targets):
                    preds_df[TARGET_COLUMNS[i]] = all_preds[:, i]

                # compute score
                preds_tensor = torch.tensor(all_preds)
                labels_tensor = torch.tensor(all_labels)

                #val_score = mcrmse_loss_fn(preds_tensor, labels_tensor)
                val_score, scores_ = MCRMSE(labels_tensor, preds_tensor)
                ave_loss = val_loss_meter.avg
                print(f"valid score = {val_score} scores= {scores_}")
                for i, ell_score in enumerate(ell_score_columns):
                    scores[ell_score] = scores_[i]

                if args.use_wandb:
                    wandb.log({"LB": val_score}, step=wandb_step)
                    for ell_score in ell_score_columns:
                        wandb.log({ell_score: scores[ell_score]}, step=wandb_step)

                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': step + 1,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': ave_loss,
                }

                if config["use_ema"]:
                    ema.restore()

                is_best = False
                for ell_score in ell_score_columns:
                    if scores[ell_score] < best_scores[ell_score]:
                        best_scores[ell_score] = scores[ell_score]

                if val_score < best_score:
                    best_score = val_score
                    is_best = True
                    tracker = 0
                else:
                    tracker += 1

                if is_best:
                    if best_score < save_trigger:
                        save_checkpoint(config, model_state, is_best=is_best)
                    preds_df.to_csv(os.path.join(config["model_dir"], f"oof_df_fold_{fold}.csv"), index=False)
                else:
                    print(f"patience reached {tracker}/{config['patience']}")

                if args.use_wandb:
                    wandb.log({"Best LB": best_score}, step=wandb_step)
                    for ell_score in ell_score_columns:
                        wandb.log({f"Best {ell_score}": best_scores[ell_score]}, step=wandb_step)

                if (config["use_awp"]) & (best_score <= config["awp_trigger"]):
                    if epoch >= config["awp_trigger_epoch"]:
                        print("AWP is triggered...")
                        AWP_FLAG = True

                if SWA_FLAG:
                    # update average model
                    if is_best:
                        model.to('cpu')
                        swa_model.update_parameters(model)
                        swa_model_name = f"swa_model_fold_{fold}"
                        swa_filename = f'{config["model_dir"]}/{swa_model_name}_model.pth.tar'
                        swa_state = {
                            'step': step + 1,
                            'state_dict': swa_model.state_dict(),
                        }
                        print("saving swa state...")
                        torch.save(swa_state, swa_filename, _use_new_zipfile_serialization=False)
                        model = accelerator.prepare(model)

                torch.cuda.empty_cache()
                model.train()
                print("GPU Utilization after evaluation...")
                print_gpu_utilization(CUDA_VISIBLE_DEVICES)
                print("=="*40)

                if tracker >= config["patience"]:
                    print("stopping early")
                    model.eval()
                    break


if __name__ == "__main__":
    run_training()
