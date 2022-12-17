import argparse
import gc
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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config_path', type=str, required=True)
    ap.add_argument('--use_wandb', action='store_true')
    args = ap.parse_args()
    return args

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

#------- Data Processing ---------------------------------------------------#


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

#--------- Main Function Helpers ------------------------------------------#


def generate_text(config, tokenizer, model, input_text, num_augment):
    test_tokenized = tokenizer.encode_plus(
        input_text,
        add_special_tokens=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=False
    )

    test_input_ids = test_tokenized["input_ids"]
    test_attention_mask = test_tokenized["attention_mask"]

    beam_outputs = model.generate(
        input_ids=test_input_ids.to("cuda"),
        attention_mask=test_attention_mask.to("cuda"),
        max_length=config["max_length"],
        early_stopping=True,
        num_beams=15,
        num_return_sequences=num_augment,
        no_repeat_ngram_size=2,
        temperature=2.0,
        do_sample=True,
    )

    to_return = []

    for beam_output in beam_outputs:
        sent = tokenizer.decode(
            beam_output,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        to_return.append(sent)
    torch.cuda.empty_cache()
    return to_return


def generate_augmentation(config, tokenizer, model, text_id, input_df, num_augment=2):
    example_df = input_df[input_df["text_id"] == text_id].copy()
    example_df = example_df.reset_index(drop=True)

    model_input = example_df["model_input"].values[0]
    augmentations = generate_text(config, tokenizer, model, model_input, num_augment=num_augment)
    for i in range(num_augment):
        example_df[f"aug_{i}"] = augmentations[i]
    return example_df

#------ Main Function --------------------------------------------------#


def run_inference(config):
    # load data
    print('loading the data...')
    ellipse_df = pd.read_csv(config["ellipse_path"])
    summary_df = pd.read_csv(config["ellipse_summary_path"])
    ellipse_df = pd.merge(
        ellipse_df,
        summary_df[["text_id", "summary_text"]],
        on="text_id",
        how="left"
    )
    print("processing the data...")
    ellipse_df = pre_process(ellipse_df)

    # TODO: load from pre-trained topic model
    print("executing the topic model...")
    ellipse_df = get_topics(ellipse_df)

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

    ellipse_df = get_target_encoding(ellipse_df)
    ellipse_df["model_input"] = ellipse_df[["target_encoded", "topic_kws", "summary_text", "snippet"]].apply(
        lambda x: get_model_input_text(x[0], x[1], x[2], x[3]), axis=1
    )
    ellipse_df["model_output"] = ellipse_df["full_text"].apply(lambda x: get_model_output_text(x))

    # create output dir
    os.makedirs(config["output_dir"], exist_ok=True)

    # tokenizer
    tokenizer = get_tokenizer(config)
    tokenizer_test(tokenizer)
    config["len_tokenizer"] = len(tokenizer)

    # model
    model = T5ForConditionalGeneration.from_pretrained(config["model_checkpoint"])
    model.resize_token_embeddings(config["len_tokenizer"])

    # load checkpoint
    ckpt = torch.load(config["ckpt_path"])
    model.load_state_dict(ckpt["state_dict"])

    del ckpt
    gc.collect()
    torch.cuda.empty_cache()

    accelerator = Accelerator()
    model = accelerator.prepare(model)
    model.eval()

    all_text_ids = ellipse_df["text_id"].tolist()
    random.shuffle(all_text_ids)
    n_essay = len(all_text_ids)

    for essay_num in tqdm(range(n_essay)):
        text_id = all_text_ids[essay_num]

        result_df = generate_augmentation(
            config, tokenizer, model, text_id, ellipse_df, num_augment=config["num_augment"]
        )
        result_df.to_csv(os.path.join(config["output_dir"], f"df_{text_id}.csv"), index=False)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path, "r") as f:
        config = json.load(f)
    run_inference(config)
