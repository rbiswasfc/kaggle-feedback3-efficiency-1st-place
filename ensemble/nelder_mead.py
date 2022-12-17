import pandas as pd
import os
import numpy as np
import scipy as sp
import argparse
from sklearn.metrics import f1_score, log_loss
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error

parser = argparse.ArgumentParser(description='nelder-mead')
parser.add_argument('-custom', action='store_true',help='Patience')
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
    #print(scores)
    return mcrmse_score

TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']

PATH = './'
FILES = os.listdir(PATH)
OOF = np.sort([f for f in FILES if 'csv' in f])
if args.custom:
    OOF = [
        #'exp001_4468_oof.csv',
        #'exp002_4484_oof.csv',
        #'exp003_4495_oof.csv',
        #'exp004_4479_oof.csv',
        #'exp006_4482_oof.csv',
        #'exp007_4496_oof.csv',
        'exp008_4473_oof.csv',
        'exp009_4462_oof.csv',
        'exp010_setfit.csv',
        'exp015_spanwise.csv',
        'exp016_v3l_t5.csv',
        'exp017_electra_pet.csv',
        'exp018_spanwise_mpl.csv',
        'exp019_spanwise_v3base.csv',
        'exp020_dexl.csv',
        'exp020a_dexl_8folds.csv',
        'exp021_v3b_kw_prefix.csv',
        'exp022_v3l_kw.csv',
        'exp022a_v3l_8folds.csv',
        #'exp100_v3b_21K_cv.4635.csv',
        'exp111_v3b_4K_cv_.4532.csv',
        'exp112_v3lb_4K_cv_.4503.csv',
        'exp113_oof_v3l_21k.csv',
        'exp114_oof_v3l_FGM_EMA.csv',
        'exp115_oof_v3l_AWP_EMA.csv',
        'exp116_oof_roberta_large_AWP_EMA.csv',
        'exp117_oof_v3l_stage2_mse_AWP_EMA.csv',
        'exp118_oof_v3l_stage2_bce_AWP_EMA.csv',
        'exp120_preprocessing.csv',
        'exp121_luke_large.csv',
        'exp130_electra_large_cv_0.4527.csv',
        'exp132_roberta_large_cv_0.4492.csv',
        'exp131_v3_large_cv_0.4471.csv',
        'exp133_longformer_large_cv_0.4497.csv',
        'exp134_deberta_large_cv_0.4510.csv',
        #'exp119_oof_v3l_stage2_mse_AWP_EMA_8fl.csv',
        'exp200-debv3b.csv',
        'exp201a-debv3b-pl.csv',
        'exp201b-debv3b-pl.csv',
        'exp201-debv3b.csv',
        'exp202a-debv3l-pl.csv',
        'exp202-debv3l.csv',
        'exp203a-debl.csv',
        'exp203-deb-l-pl.csv',
        'exp204a-pet-debv3l-pl.csv',
        'exp204b-pet-debv3l-pl.csv',
        'exp205-debv3b-pet.csv',
        'exp206-debv2xl.csv',
        'exp207a-debl-pet-8fold.csv',
        #'exp207-debv3l-pet-8fold.csv',
        'exp208-debl-8fold.csv',
        #'exp300_debv3b.csv',
        #'exp301_debv3l.csv',
        #'exp302_debl.csv',
        'exp303_debv3l.csv',
        'exp304_debl_tapt_f4.csv',
        'exp305_debv3l_mt_f4.csv',
        'exp306_dbv3b_f4.csv',
        'exp310_dbv3l_f4_pl_s2.csv',
        'exp320_dbv3l_e4_f8.csv',
        ]
    OOF = [
        'exp009_4462_oof.csv',
        #'exp018_spanwise_mpl.csv',
        'exp022a_v3l_8folds.csv',
        #'exp023_contrastive_4folds.csv',
        #'exp207a-debl-pet-8fold.csv',
        'exp112_v3lb_4K_cv_.4503.csv',
        'exp121_luke_large.csv',
        'exp132_roberta_large_cv_0.4492.csv',
        #'exp204b-pet-debv3l-pl.csv',
        'exp208-debl-8fold.csv',
        'exp306_dbv3b_f4.csv',
        'exp310_dbv3l_f4_pl_s2.csv',
        #'exp320_dbv3l_e4_f8.csv',
    ]

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
#print(base_df.shape)
#print(base_df['discourse_id'].nunique())
alloof = []
for i, tmpdf in enumerate(OOF_CSV):
    tmpdf.drop_duplicates(subset='text_id', inplace=True)
    #tmpdf.to_csv(f'tmp_{i}.csv', index=False)
    #print(tmpdf.shape)
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