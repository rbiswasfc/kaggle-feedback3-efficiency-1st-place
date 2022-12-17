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
        'exp003_4495_oof.csv',
        'exp007_4496_oof.csv',
        'exp009a_pet.csv',
        "exp013_setfit_revisit.csv",
        # 'exp022b_kw_mindist.csv',
        # "exp023_contrastive_4folds.csv",
        'exp022_v3l_kw.csv',
        'exp024_multiscale.csv',
        'exp024a_multiscale.csv',
        # 'exp025_pet_electra.csv',
        # 'exp026b_baseline_8f.csv',
        'exp026d_baseline_mindist.csv',
        'exp027_setfit_again.csv',
        'exp030_v3_small.csv',
        'exp040_ordinal.csv',
        'exp041_mpnet.csv',

        'exp120_preprocessing.csv',
        'exp121_luke_large.csv',
        'exp132_roberta_large_cv_0.4492.csv',

        'exp203b_del-mindist.csv',
        # 'exp208_debl-8fold.csv',
        # 'exp207a_debl-pet-8fold.csv',

        # "exp300_debv3b.csv",
        # "exp302_debl.csv",
        'exp303d_dbv3l_f4_dist_pl_stage2.csv',
        'exp310_dbv3l_f4_pl_s2.csv',
        # 'exp320_dbv3l_e4_f8.csv',

        'oof_svr_many_model_f4.csv',
        # 'oof_svr_many_model_f8.csv',
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
    [0.08168817649142299, -0.2522407468003228, 0.19687027922949007, 0.007164907774984664, 0.11267045826726255, 0.018678902237182306, 0.1191764679551505, -0.020833254449285528, -0.074580137774303, -0.2171641145377825, 0.3115955288269304, 0.10320505881750394, 0.23777019217820894, 0.16778241925644075, 0.16441234144198552, 0.1369378453905852, 0.036574820408480174, -0.5633039491276877, 0.41958607398859127, 0.013422568844096339],
    [-0.09980304696921537, 0.20156164118053227, -0.31690387810082343, 0.331072043986979, 0.2076616158341807, 0.21657928666151632, -0.09911603027233502, -0.15658246509679813, 0.16573271771431214, -0.09000021485272909, -0.27436552998943603, 0.2526711681694777, 0.12492742037171911, 0.24967033073987038, 0.08064657554573432, 0.07640244760146353, 0.13622941020372736, 0.09513590678413564, 0.03188056235678087, 0.009087349520710258, 0.7923984995652289, -1.0461559752850544, 0.11235487097400797],
    [-0.17457880362820777, 0.020563427940664028, -0.11242172189946065, 0.13754075475086097, 0.03404639179031507, 0.14215102745467478, 0.02835475307911739, 0.21378427632638286, 0.23246317214051482, -0.045800086489822506, -0.31721357298992725, 0.3755788367673749, 0.12768385410542338, 0.0968815397616346, -0.032126963903609834, -0.03637295831660975, 0.1546015986033874, 0.02540623876186067, 0.0006659607202642946, 0.02553103500598286, 0.006568832649322459, 0.006041932034902386, 0.09215837867819321],
    [-0.10518202236871257, 0.10528248228527909, 0.048517309196436625, 0.02583352620065557, 0.3440979798467071, 0.03297262464468234, 0.03173433659570711, 0.19933749324451616, -0.12251169559405634, -0.14474533495588146, 0.26454779243438586, 0.11731232132574936, 0.14130997974695844, -0.046412258471246554, 0.013235273024476538, 0.06617929473190454, 0.11196199322954524, -0.074217141205491, 0.08740603351477533, 0.1730494397255195, -0.3950165684075072, 0.1262506123332956],
    [0.08126794276209776, 0.25673009007458214, -0.10650879272331992, 0.3509095436152839, 0.20535361792440932, 0.01882731991846879, -0.20702708315454765, 0.04086263823683209, -0.040514462809653484, -0.1401929398272222, -0.059737004354090656, 0.1858861697103139, -0.07676033007466637, 0.24089589984330706, -0.0009034580139000766, 0.12125085418270987, 0.03427195342419255, 0.11671407902413988, 0.018127263881979747, -0.33726636471877147, 0.10769267036004573, 0.19165997924850403],
    [-0.05058688538891018, 0.12171220204276867, -0.4554223935250672, 0.3204331600106064, 0.11284478978729645, 0.036137817656626964, 0.06093308444277955, -0.07669961594144736, -0.20699400846246996, 0.20813240608483297, 0.011152267333185688, 0.1126079199678229, 0.2585708569614641, 0.059029887138994486, 0.08447378164912889, 0.24980629859432896, 0.06192284282532312, 0.003159894791544769, -0.0987596566231344, -0.027780859037410573, 0.21674219101601722],
]

    # Prune
    # cohesion
    oof_wts[0].insert(1,0)
    oof_wts[0].insert(11,0)
    oof_wts[0].insert(19,0)
    # phraseology
    oof_wts[3].insert(2,0)
    # grammar
    oof_wts[4].insert(17,0)
    # conventions
    oof_wts[5].insert(5,0)
    oof_wts[5].insert(8,0)

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
