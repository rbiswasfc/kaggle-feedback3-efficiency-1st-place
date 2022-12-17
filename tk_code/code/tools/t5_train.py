import argparse
import json
import os
import random
import re
import shutil
from dataclasses import dataclass

import nltk
import numpy as np
import pandas as pd
import torch
import torch.utils.checkpoint
import wandb
from accelerate import Accelerator
from bertopic import BERTopic
from datasets import Dataset
from nltk.corpus import stopwords
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from sklearn.feature_extraction.text import CountVectorizer
from spacy.lang.en import English
from tokenizers import AddedToken
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (DataCollatorWithPadding, T5ForConditionalGeneration,
                          T5Tokenizer, get_cosine_schedule_with_warmup)
from transformers.optimization import Adafactor

nltk.download('stopwords')


#------- Constants ----------------------------------------------------#

LABEL_COLS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

TARGET_MAP = {
    1: "terrible",
    2: "very bad",
    3: "bad",
    4: "below average",
    5: "average",
    6: "above average",
    7: "good",
    8: "very good",
    9: "excellent"
}

#------- Utils -------------------------------------------------------#


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    ap.add_argument('--use_wandb', action='store_true')
    args = ap.parse_args()
    return args


def init_wandb(config):
    project = config["project"]
    run = wandb.init(
        project=project,
        entity='kaggle-clrp',
        config=config,
        name=config['run_name'],
        anonymous="must",
        job_type="Train",
    )
    return run


#------- Training Utils ----------------------------------------------#

def save_checkpoint(config, state, is_best):
    os.makedirs(config["model_dir"], exist_ok=True)
    name = f"fpe_model_fold_{config['fold']}"

    filename = f'{config["model_dir"]}/{name}.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, f'{config["model_dir"]}/{name}_best.pth.tar')


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

#------- Tokenizer Utils ----------------------------------------------#


def tokenizer_test(tokenizer):
    print("=="*40)
    print(f"tokenizer len: {len(tokenizer)}")
    test_string = "This is a test [LF] [==3.5==] \n [==4.5==]"
    print(f"tokenizer test: {tokenizer.tokenize(test_string)}")
    print("=="*40)


def get_tokenizer(config):
    print("using auto tokenizer")
    tokenizer = T5Tokenizer.from_pretrained(
        config["model_checkpoint"],
        model_max_length=config["max_length"]
    )

    # adding new tokens
    TARGETS = ['1.0', '1.5', '2.0', '2.5', '3.0', '3.5', '4.0', '4.5', '5.0']
    NEW_TOKENS = [
        "[LF]",
    ]
    for target in TARGETS:
        NEW_TOKENS.append(f"[=={target}==]")

    # adding new tokens
    print("adding new tokens...")
    tokens_to_add = []
    for this_tok in NEW_TOKENS:
        tokens_to_add.append(AddedToken(this_tok, lstrip=False, rstrip=False))
    tokenizer.add_tokens(tokens_to_add)
    print(f"tokenizer len: {len(tokenizer)}")
    return tokenizer

#------- Main Function Helpers -------------------------------------------------#


def pre_process(df):
    # replace new line with LF
    df["full_text"] = df["full_text"].apply(lambda x: re.sub(re.compile(r'\n\n'), "[LF]", x))

    # encode target in natural language
    for target in LABEL_COLS:
        df[target] = df[target].apply(lambda s: f"{TARGET_MAP[int(2*s-1)]} [=={str(s)}==]")

    # split sentences
    nlp = English()
    nlp.add_pipe("sentencizer", config={"punct_chars": [".", "?", "!", "[LF]"]})

    def get_sample_sents(text, num_sents=2):
        """randomly select `num_sents` to form the essay snippet
        """
        doc = nlp(text)
        sents = list(doc.sents)
        sents = list(map(str, sents))
        num_sents = min(num_sents, len(sents))
        return random.sample(sents, num_sents)

    df["snippet"] = df["full_text"].apply(lambda x: " || ".join(get_sample_sents(x)))
    return df


def get_topics(df, ckpt_path=None):
    if ckpt_path is not None:
        topic_model = BERTopic.load(ckpt_path)
    else:
        sws = stopwords.words("english") + ["n't",  "'s", "'ve"]
        vectorizer_model = CountVectorizer(ngram_range=(1, 4), stop_words=sws)
        topic_model = BERTopic(vectorizer_model=vectorizer_model, min_topic_size=50, verbose=True)

    text_ids = df["text_id"].values.tolist()
    full_texts = df["full_text"].values.tolist()

    if ckpt_path is not None:
        topics, _ = topic_model.transform(full_texts)
    else:
        topics, _ = topic_model.fit_transform(full_texts)

    meta_topic = topic_model.get_topic_info()
    topic2name = dict(zip(meta_topic["Topic"], meta_topic["Name"]))

    df["topic"] = topics
    df["topic_kws"] = df["topic"].map(topic2name).apply(lambda x: "_".join(x.split("_")[1:]))
    return df


def get_dataset(df, tokenizer, config):
    keep_cols = ["model_input", "model_output"]
    df = df[keep_cols].copy()
    task_dataset = Dataset.from_pandas(df)

    try:
        task_dataset = task_dataset.remove_columns(["__index_level_0__"])
    except:
        pass

    def tokenize_function_input(examples):
        tz = tokenizer(
            examples["model_input"],
            padding=False,
            truncation=True,
            max_length=config["max_length"],
            add_special_tokens=True,
        )
        to_return = {
            "encoder_input_ids": tz["input_ids"],
            "encoder_attention_mask": tz["attention_mask"],
        }
        return to_return

    def tokenize_function_output(examples):
        tz = tokenizer(
            examples["model_output"],
            padding=False,
            truncation=True,
            max_length=config["max_length"],
            add_special_tokens=True,
        )

        to_return = {
            "decoder_input_ids": tz["input_ids"],
            "decoder_attention_mask": tz["attention_mask"],
        }
        return to_return

    task_dataset = task_dataset.map(tokenize_function_input, batched=True)
    task_dataset = task_dataset.map(tokenize_function_output, batched=True)
    return task_dataset


@dataclass
class CustomDataCollatorWithPadding(DataCollatorWithPadding):
    """
    data collector for t5 text generation
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None
    return_tensors = "pt"

    def __call__(self, features):
        # set_trace()
        encoder_features = [{"input_ids": feature["encoder_input_ids"],
                            "attention_mask": feature["encoder_attention_mask"]} for feature in features]
        decoder_features = [{"input_ids": feature["decoder_input_ids"],
                            "attention_mask": feature["decoder_attention_mask"]} for feature in features]

        encoder_batch = self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        decoder_batch = self.tokenizer.pad(
            decoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=None,
        )

        batch = dict()
        batch["encoder_input_ids"] = encoder_batch["input_ids"]
        batch["encoder_attention_mask"] = encoder_batch["attention_mask"]

        batch["decoder_input_ids"] = decoder_batch["input_ids"]
        batch["decoder_attention_mask"] = decoder_batch["attention_mask"]

        # -100 -> ignored in loss computations
        labels = []
        for ex_labels in batch["decoder_input_ids"]:
            tmp = [l if l != 0 else -100 for l in ex_labels]
            labels.append(tmp)
        batch["labels"] = labels

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


#------- Main Function  --------------------------------------------------------#


def execute_training(config):
    #----- Load dataframes -----------------------------------------------------#
    print('loading the data...')
    ellipse_df = pd.read_csv(config["ellipse_path"])
    summary_df = pd.read_csv(config["ellipse_summary_path"])
    ellipse_df = pd.merge(ellipse_df, summary_df[["text_id", "summary_text"]], on="text_id", how="left")
    fold_df = pd.read_parquet(config["fold_path"])
    ellipse_df = pd.merge(ellipse_df, fold_df, on="text_id", how="left")

    #----- Processing  ---------------------------------------------------------#
    print("processing the data...")
    ellipse_df = pre_process(ellipse_df)

    #----- Topic Modelling  ----------------------------------------------------#
    print("executing the topic model...")
    ellipse_df = get_topics(ellipse_df)

    #----- Train-Validation Split  ---------------------------------------------#
    fold = config["fold"]
    config["train_folds"] = [i for i in range(config["n_folds"]) if i != fold]
    config["valid_folds"] = [fold]

    train_df = ellipse_df[ellipse_df["kfold"].isin(config["train_folds"])].copy()
    valid_df = ellipse_df[ellipse_df["kfold"].isin(config["valid_folds"])].copy()

    print(f"shape of train_df: {train_df.shape}")
    print(f"shape of valid_df: {valid_df.shape}")
    os.makedirs(config["model_dir"], exist_ok=True)

    #----- Model Input/Output  -------------------------------------------------#

    def get_target_encoding(df):
        df["target_encoded"] = df[LABEL_COLS].apply(
            lambda x: "; ".join([f"{LABEL_COLS[i]}: {x[i]}" for i in range(len(LABEL_COLS))]), axis=1
        )
        return df

    def get_model_input_text(targets, topic_kws, summary, snippet):
        to_return = f"Task: text generation || Topic: {topic_kws} || Rating: {targets} || Summary: {summary} || Snippet: {snippet}"
        return to_return

    def get_model_output_text(full_text):
        return full_text

    train_df = get_target_encoding(train_df)
    train_df["model_input"] = train_df[["target_encoded", "topic_kws", "summary_text", "snippet"]].apply(
        lambda x: get_model_input_text(x[0], x[1], x[2], x[3]), axis=1
    )
    train_df["model_output"] = train_df["full_text"].apply(lambda x: get_model_output_text(x))

    valid_df = get_target_encoding(valid_df)
    valid_df["model_input"] = valid_df[["target_encoded", "topic_kws", "summary_text", "snippet"]].apply(
        lambda x: get_model_input_text(x[0], x[1], x[2], x[3]), axis=1
    )
    valid_df["model_output"] = valid_df["full_text"].apply(lambda x: get_model_output_text(x))
    print(train_df[["model_input", "model_output"]].sample(5))

    #----- Tokenizer ----------------------------------------------------------#
    tokenizer = get_tokenizer(config)
    tokenizer_test(tokenizer)
    config["len_tokenizer"] = len(tokenizer)

    #----- Dataset  -----------------------------------------------------------#
    print("creating the dataset...")
    train_dataset = get_dataset(train_df, tokenizer, config)
    valid_dataset = get_dataset(valid_df, tokenizer, config)

    #----- DataLoader  -----------------------------------------------------------#
    data_collector = CustomDataCollatorWithPadding(tokenizer=tokenizer)
    train_dataset.set_format(
        type=None,
        columns=['encoder_input_ids', 'encoder_attention_mask', 'decoder_input_ids', 'decoder_attention_mask']
    )

    valid_dataset.set_format(
        type=None,
        columns=['encoder_input_ids', 'encoder_attention_mask', 'decoder_input_ids', 'decoder_attention_mask']
    )

    train_dl = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=data_collector,
        pin_memory=True,
    )

    valid_dl = DataLoader(
        valid_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=data_collector,
        pin_memory=True,
    )

    #------------ Model, Optimizer, Scheduler ---------------------------------------------#
    print("creating the model...")
    model = T5ForConditionalGeneration.from_pretrained(config["model_checkpoint"])
    model.resize_token_embeddings(config["len_tokenizer"])

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    # prepare the training
    num_epochs = config["num_epochs"]
    warmup_pct = config["warmup_pct"]
    grad_accumulation_steps = config["grad_accumulation"]

    num_update_steps_per_epoch = len(train_dl)//grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct*num_training_steps)

    print(f"# updates per epoch: {num_update_steps_per_epoch}")
    print(f"# training steps: {num_training_steps}")
    print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    #-------------- Accelerator --------------------------------------------#
    accelerator = Accelerator()
    model, optimizer, train_dl, valid_dl = accelerator.prepare(
        model, optimizer, train_dl, valid_dl
    )

    #-------------- Training -----------------------------------------------#
    # wandb
    if args.use_wandb:
        print("initializing wandb run...")
        init_wandb(config)

    best_loss = 1e6
    wandb_step = 0
    tracker = 0

    for epoch in range(num_epochs):
        # close and reset progress bar
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch))
        loss_meter = AverageMeter()

        # Training
        model.train()

        for step, batch in enumerate(train_dl):
            outputs = model(
                input_ids=batch["encoder_input_ids"],
                attention_mask=batch["encoder_attention_mask"],
                labels=batch["labels"]
            )
            loss = outputs.loss
            accelerator.backward(loss)

            if (step + 1) % grad_accumulation_steps == 0:
                # take optimizer and scheduler steps
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                loss_meter.update(loss.item())

                progress_bar.set_description(
                    f"STEP: {(step+1)//grad_accumulation_steps:5}/{num_update_steps_per_epoch:5}. "
                    f"LR: {get_lr(optimizer):.4f}. "
                    f"TL: {loss_meter.avg:.4f}. "
                )

                if args.use_wandb:
                    wandb.log({"Train Loss": loss_meter.avg}, step=wandb_step)
                    wandb.log({"LR": get_lr(optimizer)}, step=wandb_step)

                progress_bar.update(1)
                wandb_step += 1

            # Evaluation
            if (step + 1) % config["eval_frequency"] == 0:
                print("\n")
                print("GPU Utilization before evaluation...")
                print("\n")
                print_gpu_utilization()

                model.eval()
                all_losses = []

                for batch_idx, batch in enumerate(valid_dl):
                    with torch.no_grad():
                        outputs = model(
                            input_ids=batch["encoder_input_ids"],
                            attention_mask=batch["encoder_attention_mask"],
                            labels=batch["labels"]
                        )
                    loss = outputs.loss
                    all_losses.append(loss.item())
                #
                ave_loss = round(np.mean(all_losses), 4)
                print(f">>> validation loss {ave_loss} <<<")

                if args.use_wandb:
                    wandb.log({"Epoch": epoch}, step=wandb_step)
                    wandb.log({"Validation Loss": ave_loss}, step=wandb_step)

                # save teacher
                accelerator.wait_for_everyone()
                model = accelerator.unwrap_model(model)
                model_state = {
                    'step': step + 1,
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'loss': ave_loss,
                }

                is_best = False
                if ave_loss < best_loss:
                    best_loss = ave_loss
                    is_best = True
                    tracker = 0
                else:
                    tracker += 1

                if is_best:
                    save_checkpoint(config, model_state, is_best=is_best)
                else:
                    print(f"patience reached {tracker}/{config['patience']}")

                torch.cuda.empty_cache()
                model.train()
                print("GPU Utilization after evaluation...")
                print_gpu_utilization()
                print("=="*40)

                if tracker >= config["patience"]:
                    print("stopping early")
                    model.eval()
                    break


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    execute_training(config)