import os
import re
from copy import deepcopy
from itertools import chain

import pandas as pd
from datasets import Dataset
from tokenizers import AddedToken
import torch
from transformers import AutoTokenizer
from loguru import logger
import textstat
import numpy as np
import readability

TEXTSTAT_COLUMNS = ['flesch_reading_ease', 'smog_index', 'automated_readability_index',
                    'dale_chall_readability_score', 'difficult_words', 'linsear_write_formula',
                    'coleman_liau_index', 'syllable_count', 'lexicon_count',
                    'monosyllabcount', 'polysyllabcount'
                    ]

READABILITY_SENTENCE_COLUMNS = [
                    'words_per_sentence', 'sentences_per_paragraph', 'type_token_ratio',
                    'words', 'paragraphs', 'long_words', 'complex_words']

READABILITY_WORD_COLUMNS = ['tobeverb','auxverb', 'conjunction', 'pronoun', 'preposition']


topic_map = {
'classes online students home': 'Is distance learning or online schooling beneficial to students?',
'advice ask people choice': 'Should you ask multiple people for advice?',
'failure success enthusiasm fail': 'Learning from failure | failure is necessary for success',
'school hours time day': 'Should school add more hours each day? | How to distribute forty work hours a week most usefully?',
'attitude positive positive attitude life': 'Having a positive attitude towards life is helpful',
'something accomplish always people': 'Idle or active? | The people accomplish more if they are always doing something, or the inactivity also serve a purpose',
'career young students high': 'Students should be committing to a career at a young age?',
'technology people use use technology': 'Does technology have a positive effect on our lives?',
'grow mastered try already': 'Unless you try to do something beyond what you have already mastered, you will never grow?',
'first impression change impressions': 'Is the first impression is impossible to change?',
'world people else accomplishment': '???',
'years school high school high': 'Is it a good idea for students to finish high school in three year and enter college or the work force one year early?',
'group working work alone': ' Which one is better: working in group or working alone?',
'play go like fun': 'What would you like to do in future?',
'activities extracurricular students school': 'Should all students participate in at least one extracurricular activity?',
'homework club homework club students': 'Should schools have an after school homework club?',
'phones cell cell phones use': 'Should students be allowed to use cell phones in school?',
'character traits choose people': 'How is your character formed?',
'older younger younger students older students': 'Pairing older students with younger students is good?',
'influence example people others': 'Should you influence others by leading by example?',
'summer break students summer break': 'Do students need summer break or they can handle not have summer but with more breaks around the year?',
'food menu lunch eat': 'Customizable menu for lunch?',
'teenagers curfew trouble law': "Do curfews keep teenagers out of trouble,or do they unfairly interfere young people's lives?",
'imagination knowledge imagine imagination important': 'Is imagination more important than knowledge?',
'selfesteem praise achievement praising': 'Praising student lead to development of self esteem?',
'classes class art take': 'Should students be required to take arts class?',
'decisions make decision people': 'Should you make your own decisions?',
'people life like want': 'Should you be yourself?'
}

#--------------- Tokenizer ---------------------------------------------#
def get_tokenizer(config):
    """load the tokenizer"""

    print("using auto tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])

    print("=="*40)
    print(f"tokenizer len: {len(tokenizer)}")
    print("=="*40)
    return tokenizer


def add_textstat_features(s):
    text = s['full_text']
    flesch_reading_ease = textstat.flesch_reading_ease(text)
    smog_index = textstat.smog_index(text)
    automated_readability_index = textstat.automated_readability_index(text)
    dale_chall_readability_score = textstat.dale_chall_readability_score(text)
    difficult_words = textstat.difficult_words(text)
    #text_standard  = textstat.text_standard(text)
    linsear_write_formula = textstat.linsear_write_formula(text)
    coleman_liau_index = textstat.coleman_liau_index(text)
    syllable_count = textstat.syllable_count(text)
    lexicon_count = textstat.lexicon_count(text)
    monosyllabcount = textstat.monosyllabcount(text)
    polysyllabcount = textstat.polysyllabcount(text)

    return flesch_reading_ease, smog_index, automated_readability_index, dale_chall_readability_score, difficult_words, \
           linsear_write_formula, coleman_liau_index, syllable_count, lexicon_count, monosyllabcount, polysyllabcount


def add_readability_features(s):
    text = s['full_text']
    results = readability.getmeasures(text, lang='en')
    metric = 'sentence info'
    values = []
    for measure in READABILITY_SENTENCE_COLUMNS:
        score = results[metric][measure]
        values.append(score)
    return values

def process_df(df):
        print("==" * 40)
        print("processing essay text and inserting new tokens at span boundaries")
        df[TEXTSTAT_COLUMNS] = df.apply(add_textstat_features, axis=1, result_type="expand")
        #df[READABILITY_SENTENCE_COLUMNS] = df.apply(add_readability_features, axis=1, result_type="expand")
        print("==" * 40)
        print(df.head())
        return df


class AuxFeedbackDataset:
    """Dataset class for feedback prize effectiveness task
    """

    def __init__(self, config):
        self.config = config
        self.load_tokenizer()

    def load_tokenizer(self):
        """load tokenizer as per config
        """
        self.tokenizer = get_tokenizer(self.config)


        if self.config["add_new_tokens"]:
            print("adding new tokens...")
            tokens_to_add = []
            for this_tok in NEW_TOKENS:
                 tokens_to_add.append(AddedToken(this_tok, lstrip=True, rstrip=False))
            self.tokenizer.add_tokens(tokens_to_add)

        print(f"tokenizer len: {len(self.tokenizer)}")



    def tokenize_function(self, examples):
        """

        Parameters
        ----------
        examples

        Returns
        -------

        """
        text = ''

        if self.config['add_prefix']:
            examples["full_text"] =  f' {self.tokenizer.sep_token} ' + examples["prefix"] + f' {self.tokenizer.sep_token} ' + examples["full_text"]

        if self.config['add_topic']:
            topic = topic_map[examples["topic"]]
            text = f'Topic: {topic}. [SEP] '


        if self.config['add_prompt']:
            #text = 'Rate the readability of the following text with a score from 1 to 5. [SEP] '
            text += 'Rate the cohesion, syntax, vocabulary, phraseology, grammar and conventions' \
                   ' for the following passage. '


        if text != '':
            text += examples["full_text"]
        else:
            text = examples["full_text"]

        #logger.debug(f'text: {text}')
        #print(text)

        tz = self.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=self.config["max_length"],
            padding=False,
            truncation=True
        )

        return tz

    def add_uids(self, examples):
        uids = []

        for uid in examples["uid"]:
            uids.append(uid)

        return {"uids": uids}

    def process_features(self, examples):
        features = []
        means = np.array(self.fts_mean)
        stds = np.array(self.fts_std)

        for example_flesch_reading_ease, example_smog_index, example_automated_readability_index, example_dale_chall_readability_score,\
            example_difficult_words, example_linsear_write_formula, example_coleman_liau_index, example_syllable_count,\
            example_lexicon_count, example_monosyllabcount, example_polysyllabcount in zip(examples["flesch_reading_ease"], examples["smog_index"], examples["automated_readability_index"],
                       examples["dale_chall_readability_score"],  examples["difficult_words"],
                       examples["linsear_write_formula"], examples["coleman_liau_index"],
                       examples["syllable_count"], examples["lexicon_count"],
                       examples["monosyllabcount"], examples["polysyllabcount"]
                       ):
            text_feature = np.array([example_flesch_reading_ease, example_smog_index, example_automated_readability_index,
                                     example_dale_chall_readability_score, example_difficult_words,
                                     example_linsear_write_formula, example_coleman_liau_index, example_syllable_count,
                                     example_lexicon_count, example_monosyllabcount, example_polysyllabcount,
                                     #example_words_per_sentence, example_sentences_per_paragraph,
                                     #example_type_token_ratio, example_words, example_paragraphs, example_long_words,
                                     #example_complex_words
                                     ])
            # Normalize
            text_feature = (text_feature - means) / stds
            features.append(text_feature)

        return {
            "features": features,
        }

    def generate_labels(self, examples):
        labels = []
        for i in range(len(examples["cohesion"])):
            labels.append([examples["cohesion"][i], examples["syntax"][i],
                           examples["vocabulary"][i], examples["phraseology"][i],
                           examples["grammar"][i],examples["conventions"][i]])

        return {"labels": labels}

    def compute_input_length(self, examples):
        return {"input_length": [len(x) for x in examples["input_ids"]]}

    def get_dataset(self, df, mode='train'):
        """main api for creating the Feedback dataset

        :param df: input annotation dataframe
        :type df: pd.DataFrame
        :param essay_df: dataframe with essay texts
        :type essay_df: pd.DataFrame
        :param mode: check if required for train or infer, defaults to 'train'
        :type mode: str, optional
        :return: the created dataset
        :rtype: Dataset
        """
        # save a sample for sanity checks
        sample_df = df.sample(min(16, len(df)))
        sample_df.to_csv(os.path.join(self.config["model_dir"], f"{mode}_df_processed.csv"), index=False)

        df = process_df(df)
        self.fts_mean = []
        self.fts_std = []
        for col in TEXTSTAT_COLUMNS:
            mean = df[col].mean()
            std = df[col].std()
            #logger.debug(f'Col: {col} mean: {mean} std: {std}')
            self.fts_mean.append(mean)
            self.fts_std.append(std)

        task_dataset = Dataset.from_pandas(df)
        task_dataset = task_dataset.map(self.tokenize_function, batched=False)
        task_dataset = task_dataset.map(self.compute_input_length, batched=True)
        task_dataset = task_dataset.map(self.process_features, batched=True)

        if mode != "infer":
            task_dataset = task_dataset.map(self.generate_labels, batched=True)
        try:
            task_dataset = task_dataset.remove_columns(column_names=["__index_level_0__"])
        except Exception as e:
            pass
        return df, task_dataset, self.tokenizer

#--------------- dataset with truncation ---------------------------------------------#

def get_dataset(config, df, mode="train"):
    """Function to get fast approach dataset with truncation & sliding window
    """
    dataset_creator = AuxFeedbackDataset(config)
    _, task_dataset, tokenizer = dataset_creator.get_dataset(df, mode=mode)

    to_return = dict()
    to_return["dataset"] = task_dataset
    to_return["tokenizer"] = tokenizer
    return to_return
