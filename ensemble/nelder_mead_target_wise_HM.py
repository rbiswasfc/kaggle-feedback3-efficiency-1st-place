import argparse
import os
from functools import partial

import numpy as np
import pandas as pd
import scipy as sp
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss, mean_squared_error

yao_train_path = "/Users/heyao/kaggle/feedback-ells/input/feedback-prize-english-language-learning/train.csv"
tk_train_path = "../../datasets/feedback-prize-english-language-learning/train.csv"
parser = argparse.ArgumentParser(description='nelder-mead')
parser.add_argument('-train_path', type=str, default=tk_train_path, required=False, help="train csv path")
parser.add_argument('-custom', action='store_true', help='Patience')
parser.add_argument('-exclude', type=str, nargs='+', help='List of oof csvs', required=False)

args = parser.parse_args()

# YAO: modify get_score, only pass one target once.


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
    OOF = ["exp009_4462_oof.csv",
        "exp022a_v3l_8folds.csv",
        "exp121_luke_large.csv",
        "exp132_roberta_large_cv_0.4492.csv",
        "exp132a_roberta_large_8_fl_cv_0.4487.csv",
        "exp117d_v3_large_1428_cv_0.4475.csv",
        "exp208_debl-8fold.csv",
        "exp306_dbv3b_f4.csv",
        "exp310_dbv3l_f4_pl_s2.csv"
    ]
    """
    OOF = [
        './level2/lgb_ensemble.csv',
        #'./level2/ens33_oof_lstm_9m.csv',
        './level2/ens42_meta_lstm.csv',
        #'./level2/ensemble_df.csv'
    ]
    """
    OOF = np.array(OOF)

excludes = args.exclude
if excludes:
    for exl in excludes:
        idx = np.where(OOF == exl)[0][0]
        OOF = np.delete(OOF, idx)

try:
    train_df = pd.read_csv(args.train_path)
except:
    train_df = pd.read_csv("C:\\Users\\mehta\\Desktop\\kaggle\\feedback_phr_se\\train.csv")
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


# YAO: add arg `i`
def min_func(K, i):
    ypredtrain = 0
    for a in range(len(alloof)):
        ypredtrain += K[a] * alloof[a]
    return get_score(TRUE[:, i], ypredtrain[:, i])


# YAO: add for loop to get single target best weight
scores = []
weights = []
for i in range(6):
    res = minimize(partial(min_func, i=i), [1 / len(alloof)] * len(alloof), method='Nelder-Mead', tol=1e-6)
    K = res.x
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

    score = get_score(TRUE[:, i], ypredtrain[:, i])
    print(f'Score: {score}')
    print(f'Wt sum: {np.sum(K)}')
    print(f"{TARGET_COLUMNS[i] + ' end':=^60}")
    scores.append(score)
    weights.append(oof_wts)
print(f"Overall MCRMSE: {np.mean(scores)}")
print("oofs:")
for i in range(len(OOF.tolist())):
    # print(OOF.tolist())
    oof_name = OOF.tolist()[i].split('_')[0]
    print(f'{oof_name}_df,')
print("weights:")
print("[")
for weight in weights:
    print(weight, end="")
    print(",")
print("]")
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
