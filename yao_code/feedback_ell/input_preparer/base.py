"""
@created by: heyao
@created at: 2022-08-24 23:50:13
"""
from typing import List

import pandas as pd
import transformers
from omegaconf import DictConfig

from feedback_ell.utils.preprocessing import resolve_encodings_and_normalize


class BaseInputPreparer(object):
    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, config: DictConfig):
        # Yes, I bind the config with all the classes for simply interface.
        self.tokenizer = tokenizer
        self.config = config
        self.label_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    def preprocess_df(self, df):
        if self.config.train.clean:
            df["full_text"] = df["full_text"].apply(resolve_encodings_and_normalize)
        return df

    def prepare_input(self, df: pd.DataFrame) -> [List, List, List]:
        raise NotImplementedError()
