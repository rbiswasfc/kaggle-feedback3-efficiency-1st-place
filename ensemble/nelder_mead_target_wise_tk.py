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
parser.add_argument('-recheck', action='store_true', help='Patience')
parser.add_argument('-method', type=str, default='nm', required=False)

args = parser.parse_args()

method_map = {
    'nm': 'Nelder-Mead', # ok
    'slsqp': 'SLSQP', # fast
    'powell': 'Powell', # slow
    'cg': 'CG', # good
    'bfgs': 'BFGS', #best so far
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
#TARGET_COLUMNS = ['cohesion']

PATH = './'
FILES = os.listdir(PATH)
OOF = np.sort([f for f in FILES if 'csv' in f])
if args.custom:
    OOF = [
        #'exp003_4495_oof.csv',
        #'exp004_4479_oof.csv',
        'exp009a_pet.csv',
        #"exp013_setfit_revisit.csv",
        'exp022b_kw_mindist.csv',
        'exp024_multiscale.csv',
        'exp024a_multiscale.csv',
        'exp026b_baseline_8f.csv',
        'exp027_setfit_again.csv',
        'exp030_v3_small.csv',
        #'exp031_v3_small_balanced.csv',
        #'exp040_ordinal.csv',
        #'exp041_mpnet.csv',
        #'exp120_preprocessing.csv',
        #'exp121_luke_large.csv',
        'exp132_roberta_large_cv_0.4492.csv',
        #'exp137_xlm_roberta_large_cv_0.4535.csv',
        'exp203b_del-mindist.csv',
        'exp208_debl-8fold.csv',
        'exp207a_debl-pet-8fold.csv',
        "exp300_debv3b.csv",
        "exp302_debl.csv",
        'exp303d_dbv3l_f4_dist_pl_stage2.csv',
        'exp310_dbv3l_f4_pl_s2.csv',
        'exp320_dbv3l_e4_f8.csv',
        'oof_svr_many_model_f4.csv',
        'oof_svr_many_model_f8.csv',
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

train_df = pd.read_csv(args.train_path)
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


if not args.recheck:
    # YAO: add for loop to get single target best weight
    scores = []
    weights = []
    for i in range(6):
        if args.method == 'bfgs':
            res = minimize(
                partial(min_func, i=i), [1 / len(alloof)] * len(alloof),
                method= method_map[args.method],
                tol = 1e-10,
        )
        else:
            res = minimize(
                    partial(min_func, i=i), [1 / len(alloof)] * len(alloof),
                    constraints=({'type': 'eq','fun': lambda w: 1-sum(w)}),
                    method= method_map[args.method],
                    options = {'ftol':1e-10},
                    #bounds=[(-1.0, 1.0)] * len(alloof),
            )
        K = res.x
        #K = [0.4, 0.6]
        #print(K)

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
else:
    #"""
    # Override wts here
    #"""
    oof_wts = [
    [
    0.0,0.2293,0.0,0.0,0.0,0.09405,0.0,0.0,0.18386,0.0,0.10751,0.10983,0.12221,0.14324,0.0,0.01,0.0,0.0
    ],   
    [
    0.10594,0.2045,0.12928,0.02895,0.0,0.0,0.07559,0.0468,0.13633,0.09111,0.0,0.08955,0.05696,0.035,0.0,0.0,0.0,0.0
    ],
    [
    0.0,0.0,0.0,0.08643,0.0,0.24342,0.15523,0.09613,0.17627,0.0,0.0,0.0,0.10951,0.0,0.0297,0.01,0.0,0.09333
    ],
    [
    0.0,0.0,0.02,0.17271,0.1911,0.0294,0.14416,0.10974,0.13913,0.02852,0.0,0.0,0.04426,0.08409,0.0,0.03688,0.0,0.0
    ],
    [
    0.22439,0.20302,0.13359,0.0,0.0,0.0,0.0,0.06518,0.0,0.10171,0.0,0.09066,0.0,0.0,0.11257,0.025,0.0,0.04388
    ],
    [
    0.00995,0.27011,0.005,0.0,0.0,0.0,0.0,0.01478,0.0,0.06718,0.14155,0.03803,0.07606,0.20377,0.07303,0.01941,0.0,0.08113
    ]
]

    # Prune
    # cohesion
    #oof_wts[0].insert(1,0)
    #oof_wts[0].insert(11,0)
    #oof_wts[0].insert(19,0)
    # phraseology
    #oof_wts[3].insert(2,0)
    # grammar
    #oof_wts[4].insert(17,0)
    # conventions
    #oof_wts[5].insert(5,0)
    #oof_wts[5].insert(8,0)

    # recheck values
    scores = []
    for i in range(6):
        oof_files = []
        ypredtrain =0
        for a in range(len(alloof)):
            print(OOF[a], oof_wts[i][a])
            oof_files.append(OOF[a])
            ypredtrain +=  oof_wts[i][a] * alloof[a]

        score = get_score(TRUE[:, i], ypredtrain[:, i])
        print(f'Score: {score}')
        print(f'Wt sum: {np.sum(oof_wts[i])}')
        print(f"{TARGET_COLUMNS[i] + ' end':=^60}")
        scores.append(score)
    print(f"Overall MCRMSE: {np.mean(scores)}")
    for i in range(len(OOF.tolist())):
        # print(OOF.tolist())
        oof_name = OOF.tolist()[i].split('_')[0]
        print(f'{oof_name}_df,')
    print("weights:")
    print("[")
    for weight in oof_wts:
        print(weight, end="")
        print(",")
    print("]")

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
