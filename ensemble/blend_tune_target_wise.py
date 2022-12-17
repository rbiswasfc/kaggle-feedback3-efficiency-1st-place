"""
@created by: heyao
@created at: 2022-10-31 20:30:00
Modify Trushant's code which used in NBME comp :)
"""
import time
from functools import partial

import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def weighted_blend(oofs, weights=None):
    weights = weights or [1 / len(oofs)] * len(oofs)
    preds = np.sum([oof * w for oof, w in zip(oofs, weights)], axis=0)
    return preds


def objective(trial: optuna.Trial, oofs, targets):
    weights = [trial.suggest_float(f"w{i}", low=0.05, high=0.6) for i in range(len(oofs))]
    weights = np.array(weights)
    weights = weights / weights.sum()
    weights = weights.tolist()
    blend = weighted_blend(oofs, weights=weights)
    # print(targets.max())
    # print(blend.max())
    score = mean_squared_error(targets, blend, squared=False)
    return score


def tune(train_df, oofs, oof_names=None, blend_on_target=None):
    """Tuning weight with different target.
    target-wise weight tuning.

    Args:
        train_df: pd.DataFrame.
        oofs: [pd.DataFrame, pd.DataFrame, ...]. list of oof dataframe.
        oof_names: list of str.
        blend_on_target: None or list of int from 0 to 5. blend on `blend_on_target` targets.

    Returns: [weights, score]. weights of oofs. (in same order), competition_score

    """
    # data prepare
    blend_on_target = blend_on_target if blend_on_target is not None else [0, 1, 2, 3, 4, 5]
    oof_names = oof_names if oof_names is not None else [f"oof{i}" for i in range(len(oofs))]
    assert len(oofs) == len(oof_names), "oofs and oof_names should be same length"

    label_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
    blend_target = [label_columns[i] for i in blend_on_target]
    oofs = [oof.loc[:, blend_target] for oof in oofs]

    target = train_df.loc[:, blend_target].values

    # tuning
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=0,
        interval_steps=1,
    )
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
        load_if_exists=False
    )
    start_time = time.perf_counter()
    study.optimize(
        partial(objective, oofs=oofs, targets=target),
        n_trials=100,
        timeout=None,
        gc_after_trial=True,
        n_jobs=6
    )
    best_params = study.best_params
    best_score = study.best_trial.values
    end_time = time.perf_counter()
    weights = [[best_params[f"w{i}"] for i in range(len(best_params))]]
    weights = np.array(weights)
    weights = weights / weights.sum()
    weights = weights[0].tolist()
    blend = weighted_blend([oof.loc[:, blend_target].values for oof in oofs], weights=weights)
    for i, target_idx in enumerate(blend_on_target):
        print(label_columns[target_idx], mean_squared_error(target[:, i], blend[:, i], squared=False))
    print(f'Best score: {best_score}, {dict(zip(oof_names, weights))}')
    print(f'Total time: {end_time - start_time}')
    print("=" * 60)
    return weights, float(best_score[0])


if __name__ == '__main__':
    import argparse
    import os

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    label_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]

    parser = argparse.ArgumentParser(description='optuna for target-wise tune')
    parser.add_argument('-train_path', type=str, help='path to fold data for csvs', required=True)
    parser.add_argument('-oof_dir', type=str, default='./', help='path to dir containing oof csvs', required=False)
    # parser.add_argument('-tol', type=float, default=0.0003, help='Tolerance. Decrease to try more models')
    # parser.add_argument('-pat', type=int, default=10, help='Patience')
    parser.add_argument('-custom', action='store_true', help='use custom oof_filenames in code.')
    parser.add_argument('-custom_oof', type=str, nargs='+',
                        help='use custom oof_filenames in command line.', required=False)
    parser.add_argument('-exclude', type=str, nargs='+', help='List of oof csvs', required=False)
    parser.add_argument('-filter', type=str, help='filter for csvs', required=False)
    args = parser.parse_args()

    # Read OOFs
    PATH = args.oof_dir  # './'
    FILES = os.listdir(PATH)
    if args.filter:
        OOF = np.sort([f for f in FILES if args.filter in f])
    else:
        OOF = np.sort([f for f in FILES if 'csv' in f])

    if args.custom:
        OOF = [
            'exp009_4462_oof.csv',
            'exp208-debl-8fold.csv',
            'exp303_debv3l.csv',
            'exp306_dbv3b_f4.csv',
            'exp015_spanwise.csv',
            'exp132_roberta_large_cv_0.4492.csv',
        ]  # 0.442796 HC tol=0.0001 all -> 0.442431

    excludes = args.exclude
    if excludes:
        for exl in excludes:
            print(exl)
            idx = np.where(OOF == exl)[0][0]
            OOF = np.delete(OOF, idx)

    custom_oofs = args.custom_oof
    if custom_oofs:
        OOF = custom_oofs

    train_df = pd.read_csv(args.train_path)
    train_df = train_df.sort_values(by='text_id', ascending=True).reset_index(drop=True)

    OOF_CSV = [pd.read_csv(name).sort_values(by='text_id', ascending=True).reset_index(drop=True) for name in OOF]
    weights, scores = [], []
    for target_idx in range(6):
        weight, score = tune(train_df, OOF_CSV, oof_names=OOF, blend_on_target=[target_idx])
        weights.append(weight)
        scores.append(score)
    print(f"OOFS: {OOF}")
    for target_name, weight, score in zip(label_columns, weights, scores):
        print(target_name, score, weight)
    print(f"Overall CV: {np.mean(scores)}")
