"""
@created by: heyao
@created at: 2022-10-14 22:30:59
"""
import pandas as pd
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from feedback_ell.input_preparer.external_tags import ExternalTagRegressionInputPreparer
from feedback_ell.input_preparer.regression import NoParagraphOrderRegressionInputPreparer, TokenRegressionInputPreparer

f = "/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv"
df = pd.read_csv(f)
config = OmegaConf.load("../../config/deberta_v3_large_reg.yaml")
config.train.external_tag_filename = "/home/heyao/kaggle/feedback-ells/input/deps.pkl"
model_path = config.model.path
tokenizer = AutoTokenizer.from_pretrained(model_path)
input_preparer = TokenRegressionInputPreparer(tokenizer, config)
encodings, labels, text_ids = input_preparer.prepare_input(df)
print(len(encodings[0]), len(labels[0]), len(text_ids))
print(labels[0])
print(encodings[0])
print(tokenizer.decode(encodings[0]["input_ids"]))
