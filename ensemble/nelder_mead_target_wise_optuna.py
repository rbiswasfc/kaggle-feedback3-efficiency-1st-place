import pandas as pd
import os
import numpy as np
import scipy as sp
import argparse
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
from optuna import Trial
from optuna.samplers import TPESampler
import optuna
from functools import partial

parser = argparse.ArgumentParser(description='nelder-mead')
parser.add_argument('-custom', action='store_true',help='Patience')
parser.add_argument('-exclude', type=str, nargs='+', help='List of oof csvs', required=False)
parser.add_argument('-n_trials', type=int, default=10)
parser.add_argument('-num_exp', type=int, default=15)
parser.add_argument('-method', type=str, default='nm', required=False)

args = parser.parse_args()

method_map = {
    'nm': 'Nelder-Mead',
    'slsqp': 'SLSQP', # fast
    'powell': 'Powell', # slow
    'cg': 'CG', # good
    'bfgs': 'BFGS', #good
    'ncg': 'Newton-CG', #fails
    'lbfgs': 'L-BFGS-B' , #good
    'tnc': 'TNC', # not so good
    'cob': 'COBYLA' , #fails
    'dog': 'dogleg' , #fails
    'tc': 'trust-constr',
    'tncg': 'trust-ncg' ,
    'te': 'trust-exact' ,
    'tk': 'trust-krylov' ,
}

def get_score(y_trues, y_preds):
    #print(y_trues, y_preds)
    # scores = []
    # idxes = y_trues.shape[1]
    # for i in range(idxes):
    #     y_true = y_trues[:, i]
    #     y_pred = y_preds[:, i]
    score = mean_squared_error(y_trues, y_preds, squared=False)
    # scores.append(score)
    # mcrmse_score = np.mean(scores)
    # print(scores)
    return score

TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

PATH = './'
FILES = os.listdir(PATH)
OOF = np.sort([f for f in FILES if 'csv' in f])

if args.custom:
        OOF= [
        'exp007_4496_oof.csv',
        'exp009_4462_oof.csv',
        'exp009a_pet.csv',
        "exp013_setfit_revisit.csv",
        'exp022a_v3l_8folds.csv',
        'exp023_contrastive_4folds.csv',
        'exp024_multiscale.csv',
        'exp024a_multiscale.csv',
        'exp026b_baseline_8f.csv',
        'exp026d_baseline_mindist.csv',
        'exp027_setfit_again.csv',
        'exp030_v3_small.csv',
        #'exp032_v3_decomposed_384.csv',
        'exp041_mpnet.csv',
        'exp120_preprocessing.csv',
        #'exp120_preprocessing.csv',
        'exp121_luke_large.csv',
        #'exp121_luke_large.csv',
        'exp132_roberta_large_cv_0.4492.csv',
        #'exp132_roberta_large_cv_0.4492.csv',
        'exp137_xlm_roberta_large_cv_0.4535.csv',
        'exp203b_del-mindist.csv',
        'exp207a_debl-pet-8fold.csv',
        'exp208_debl-8fold.csv',
        'exp300_debv3b.csv',
        'exp302_debl.csv',
        'exp303d_dbv3l_f4_dist_pl_stage2.csv',
        'exp304_debl_tapt_f4.csv',
        'exp306_dbv3b_f4.csv',
        'exp310_dbv3l_f4_pl_s2.csv',
        'exp320_dbv3l_e4_f8.csv',
        'oof_svr_many_model_f4.csv',
        'oof_svr_many_model_f8.csv'
    ]

excludes = args.exclude
if excludes:
    for exl in excludes:
        idx = np.where(OOF == exl)[0][0]
        OOF = np.delete(OOF, idx)

train_df = pd.read_csv('../../datasets/feedback-prize-english-language-learning/train.csv')
train_df = train_df.sort_values(by='text_id').reset_index(drop=True)
TRUE = train_df[TARGET_COLUMNS].values


def objective(trial: Trial) -> float:
    import random

    from collections import defaultdict

    params = defaultdict()
    for i in range(args.num_exp):
        params[f'oof{i}'] = trial.suggest_categorical(f'oof{i}', OOF)

    OOF_CSV_NM = [f"{params[f'oof{i}']}" for i in range(args.num_exp)]
    #print(OOF_CSV_NM)
    OOF_CSV_VALS = [pd.read_csv(PATH + k).sort_values(by='text_id', ascending=True).reset_index(drop=True) for k in OOF_CSV_NM]

    alloof = []
    for i, tmpdf in enumerate(OOF_CSV_VALS):
        tmpdf.drop_duplicates(subset='text_id', inplace=True)
        #tmpdf.to_csv(f'tmp_{i}.csv', index=False)
        # print(tmpdf.shape)
        mpred = tmpdf[TARGET_COLUMNS].values
        alloof.append(mpred)

    def min_func(K, i):
        ypredtrain = 0
        for a in range(len(alloof)):
            ypredtrain += K[a] * alloof[a]
        return get_score(TRUE[:, i], ypredtrain[:, i])

    # YAO: add for loop to get single target best weight
    scores = []
    weights = []
    for i in range(6):
        res = minimize(
                partial(min_func, i=i), [1 / len(alloof)] * len(alloof),
                #constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                method= method_map[args.method],
                tol=1e-10,
                #options = {'ftol':1e-10},
                #bounds=[(-1.0, 1.0)] * len(alloof),
        )        
        K = res.x
        #K = [0.4, 0.6]
        #print(K)

        ypredtrain = 0

        oof_files = []
        oof_wts = []

        for a in range(len(alloof)):
            #print(OOF[a], K[a])
            oof_files.append(OOF_CSV_NM[a])
            oof_wts.append(K[a])
            ypredtrain += K[a] * alloof[a]

        score = get_score(TRUE[:, i], ypredtrain[:, i])
        #print(f'Score: {score}')
        #print(f'Wt sum: {np.sum(K)}')
        #print(f"{TARGET_COLUMNS[i] + ' end':=^60}")
        scores.append(score)
        weights.append(oof_wts)
    print(f"Overall MCRMSE: {np.mean(scores)}")
    #print("oofs:")
    for i in range(len(OOF_CSV_NM)):
            # print(OOF.tolist())
            oof_name = OOF_CSV_NM[i].split('_')[0]
            #print(f'{oof_name}_df,')
    #print("weights:")
    #print("[")
    #for weight in weights:
    #        print(weight, end="")
    #        print(",")
    #print("]")
    return np.mean(scores)



study = optuna.create_study(direction='minimize', sampler=TPESampler())
study.optimize(lambda trial: objective(trial), n_trials=args.n_trials)
#print(study.best_trial)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))

