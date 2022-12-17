import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import textstat
import numpy as np

def create_cv_folds(df, n_splits=4, seed=461):
    """create cross validation folds
    :param df: input dataframe of labelled data
    :type df: pd.DataFrame
    :param n_splits: how many cross validation splits to perform, defaults to 5
    :type n_splits: int, optional
    :return: dataframe with kfold column added
    :rtype: pd.DataFrame
    """
    df["kfold"] = -1

    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    data_labels = df[labels].values

    for f, (t_, v_) in enumerate(mskf.split(df, data_labels)):
        df.loc[v_, "kfold"] = f
    return df

def add_fts(data):
    data['text_standard'] = data['full_text'].apply(lambda x: textstat.automated_readability_index(x))
    counts = data['text_standard'].value_counts().reset_index()
    counts.to_csv('counts.csv', index=False)
    return data

def create_folds_yao(data, num_splits, add_std=True):
    labels = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    data["kfold"] = -1
    data["row_std"] = data[labels].apply(lambda x: round(np.std(x), 1), axis=1)
    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=42)
    if add_std:
        labels += ["row_std"]
    data_labels = data[labels].values

    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f

    data.drop(columns=['row_std'], inplace=True)
    return data

data = pd.read_csv("../datasets/feedback-prize-english-language-learning/train.csv")
#data = add_fts(data)
for fold in [4,5,8,10]:
    data = create_cv_folds(data, n_splits=fold)
    data.to_csv(f"folds/train_{fold}folds.csv", index=False)

#folds = data.groupby(['kfold'])['text_standard'].value_counts()
#print(folds)
#folds.to_csv('folds.csv', index=False)
print("Folds created successfully")