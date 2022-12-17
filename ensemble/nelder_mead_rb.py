import argparse
import os

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss, mean_squared_error

parser = argparse.ArgumentParser(description='nelder-mead')
parser.add_argument('-custom', action='store_true', help='Patience')
parser.add_argument('-exclude', type=str, nargs='+', help='List of oof csvs', required=False)

args = parser.parse_args()


def get_score(y_trues, y_preds):
    #print(y_trues, y_preds)
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)
        scores.append(score)
    mcrmse_score = np.mean(scores)
    # print(scores)
    return mcrmse_score


TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

PATH = './'
FILES = os.listdir(PATH)
OOF = np.sort([f for f in FILES if 'csv' in f])
if args.custom:
    OOF = [
        # 'exp002_4484_oof.csv',
        # 'exp003_4495_oof.csv',
        # 'exp004_4479_oof.csv',
        # 'exp006_4482_oof.csv',
        'exp007_4496_oof.csv',
        'exp008_4473_oof.csv',
        # 'exp010_setfit.csv',
        'exp009_4462_oof.csv',
        # 'exp011_aux.csv',
        # 'exp012_corrector.csv',
        # 'exp013_setfit_revisit.csv',
        # 'exp014_resolved.csv',
        'exp015_spanwise.csv',
        # 'exp015_spanwise_pl.csv',
        # 'exp015c_spanwise_8folds.csv',
        # 'exp016_v3l_t5.csv',
        # 'exp018_spanwise_mpl.csv',
        'exp022a_v3l_8folds.csv',
        'exp020a_dexl_8folds.csv',
        'exp023_contrastive_4folds.csv',
        "exp028_baseline_targets.csv",
    ]
    """
    OOF = [
        './level2/lgb_ensemble.csv',
        #'./level2/ens33_oof_lstm_9m.csv',
        './level2/ens42_meta_lstm.csv',
        #'./level2/ensemble_df.csv'
    ]
    """

excludes = args.exclude
if excludes:
    for exl in excludes:
        idx = np.where(OOF == exl)[0][0]
        OOF = np.delete(OOF, idx)

train_df = pd.read_csv('../../datasets/feedback-prize-english-language-learning/train.csv')
train_df = train_df.sort_values(by='text_id').reset_index(drop=True)
TRUE = train_df[TARGET_COLUMNS].values

OOF_CSV = [pd.read_csv(PATH + k).sort_values(by='text_id', ascending=True).reset_index(drop=True) for k in OOF]

#base_df = pd.read_csv('oof_a-delv3-prod-8-folds.csv')
# print(base_df.shape)
# print(base_df['discourse_id'].nunique())
alloof = []
for i, tmpdf in enumerate(OOF_CSV):
    tmpdf.drop_duplicates(subset='text_id', inplace=True)
    #tmpdf.to_csv(f'tmp_{i}.csv', index=False)
    # print(tmpdf.shape)
    mpred = tmpdf[TARGET_COLUMNS].values
    alloof.append(mpred)


def min_func(K):
    ypredtrain = 0
    for a in range(len(alloof)):
        ypredtrain += K[a] * alloof[a]
    return get_score(TRUE, ypredtrain)


res = minimize(min_func, [1 / len(alloof)] * len(alloof), method='Nelder-Mead', tol=1e-6)
K = res.x
# Override wts here
"""
K = [0.11, # exp7 - dexl
     0.07, # exp8 - kd
     0.11, # exp10 - debv3-l 8 fold
     0.10, # exp16 - debv3-l 10 fold
     0.20, # exp209 - debv3-l
     0.11, # exp212 - lf
     0.24, # exp213 - deb-l
     0.06  # exp214 - v2-xl
     ]
"""
#K = [0.4, 0.6]
print(K)

ypredtrain = 0

oof_files = []
oof_wts = []

for a in range(len(alloof)):
    print(OOF[a], K[a])
    oof_files.append(OOF[a])
    oof_wts.append(K[a])
    ypredtrain += K[a] * alloof[a]


score = get_score(TRUE, ypredtrain)
print(f'Score: {score}')
print(f'Wt sum: {np.sum(K)}')
# recheck values

"""
ens_df = pd.DataFrame(data={'oof_file': oof_files, 'wt': oof_wts})
ens_df = ens_df.sort_values(by='wt', ascending=False).reset_index(drop=True)
print(ens_df)

ensemble_df = pd.DataFrame()
ensemble_df['text_id'] = train_df['text_id']
ensemble_df['label'] = TRUE
ensemble_df = pd.concat([ensemble_df, pd.DataFrame(ypredtrain)], axis=1)
ensemble_df.rename(columns={0: "Ineffective", 1:"Adequate", 2:"Effective"}, inplace=True)
os.makedirs('./level2', exist_ok=True)
#ensemble_df.to_csv('./level2/ensemble_df.csv', index=False)
"""
