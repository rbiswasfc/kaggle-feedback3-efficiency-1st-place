"""
@created by: heyao
@created at: 2022-09-05 00:20:02
copy and edit from: https://www.kaggle.com/code/abhishek/multi-label-stratified-folds/notebook
"""
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.model_selection import StratifiedKFold

pd.options.display.max_columns = 100


def create_folds(data, num_splits, add_std=True, random_state=36):
    labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    data["kfold"] = -1
    data["row_std"] = data[labels].apply(lambda x: round(np.std(x), 1), axis=1)
    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    if add_std:
        labels += ["row_std"]
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f

    return data


def make_folds(data, num_splits, add_std=True):
    data["fold"] = -1
    mskf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=36)
    for f, (t_, v_) in enumerate(mskf.split(data, data.cluster)):
        data.loc[v_, "kfold"] = f

    return data

if __name__ == '__main__':
    data = pd.read_csv("/home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning/train.csv")
    data = create_folds(data, num_splits=5, add_std=True)
    data.to_csv("/home/heyao/kaggle/feedback-ells/input/fb3_folds.csv", index=False)
    # print(data.describe())
    labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    for label in labels:
        print(data[label].value_counts(normalize=True))
    print("=" * 60)
    for fold in range(5):
        # print(data[data.fold == fold].describe())
        for label in labels:
            print(data.loc[data.kfold == fold, label].value_counts(normalize=True))
        print("=" * 60)
    # folds = pd.read_csv("/home/heyao/kaggle/feedback-ells/input/fb3_folds11.csv")
    # print((data["fold"] != folds.fold).sum())
