import os
import sys
from pathlib import Path

import hydra
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from omegaconf import DictConfig, OmegaConf


def print_line():
    prefix, unit, suffix = "#", "--", "#"
    print(prefix + unit*50 + suffix)


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


@hydra.main(version_base=None, config_path="../conf/processing", config_name="cv_mapping")
def main(cfg):
    # config
    print_line()
    print("config used for fold split...")
    print(OmegaConf.to_yaml(cfg))
    print_line()

    # crete folder to keep fold data
    os.makedirs(cfg.outputs.output_dir, exist_ok=True)

    # create folds
    data_dir = cfg.competition_dataset.data_dir
    train_path = cfg.competition_dataset.train_path
    train_df = pd.read_csv(os.path.join(data_dir, train_path))
    train_df = create_cv_folds(train_df, n_splits=cfg.fold_metadata.n_folds)
    fold_df = train_df[["text_id", "kfold"]].drop_duplicates()
    fold_df = fold_df.reset_index(drop=True)
    fold_df.to_parquet(os.path.join(cfg.outputs.output_dir, cfg.outputs.fold_path))
    print(fold_df.head())


if __name__ == "__main__":
    main()
