#!/usr/bin/env python

import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']


def get_score(y_trues, y_preds):
    #print(y_trues, y_preds)
    scores = []
    idxes = y_trues.shape[1]
    for i in [4]:  # range(idxes):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)
        scores.append(score)
    mcrmse_score = np.mean(scores)
    # print(scores)
    return mcrmse_score


def main():
    # arguments
    parser = argparse.ArgumentParser(description='hill climbing')
    parser.add_argument('-train_path', type=str, help='path to train data for csvs', required=True)
    parser.add_argument('-oof_dir', type=str, default='./', help='path to dir containing oof csvs', required=False)
    parser.add_argument('-tol', type=float, default=0.0003, help='Tolerance. Decrease to try more models')
    parser.add_argument('-pat', type=int, default=10, help='Patience')
    parser.add_argument('-custom', action='store_true', help='Patience')
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
            'exp003_4495_oof.csv',
            'exp007_4496_oof.csv',
            'exp009a_pet.csv',
            "exp013_setfit_revisit.csv",
            'exp022b_kw_mindist.csv',
            "exp023_contrastive_4folds.csv",
            'exp024_multiscale.csv',
            'exp024a_multiscale.csv',
            # 'exp025_pet_electra.csv',
            'exp026b_baseline_8f.csv',
            # 'exp026d_baseline_mindist.csv',
            'exp027_setfit_again.csv',
            'exp030_v3_small.csv',
            'exp040_ordinal.csv',
            'exp041_mpnet.csv',

            'exp120_preprocessing.csv',
            'exp121_luke_large.csv',
            'exp132_roberta_large_cv_0.4492.csv',

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

    excludes = args.exclude
    if excludes:
        for exl in excludes:
            print(exl)
            idx = np.where(OOF == exl)[0][0]
            OOF = np.delete(OOF, idx)

    print(OOF)
    OOF_CSV = []
    OOF_CSV = [pd.read_csv(PATH + k).sort_values(by='text_id', ascending=True).reset_index(drop=True) for k in OOF]

    print('We have %i oof files...' % len(OOF))

    # 6 labels
    x = np.zeros((len(OOF_CSV[0]), 6, len(OOF)))
    for k in range(len(OOF)):
        oof_df = OOF_CSV[k]
        #oof_df.drop_duplicates(subset='text_id', inplace=True)
        values = oof_df[TARGET_COLUMNS].values
        x[:, :, k] = values

    # print(x.shape)
    # train_df = pd.read_csv('../../datasets/feedback-prize-english-language-learning/train.csv')
    train_df = pd.read_csv(args.train_path)

    train_df = train_df.sort_values(by='text_id', ascending=True).reset_index(drop=True)
    TRUE = train_df[TARGET_COLUMNS].values

    all = []
    for k in range(x.shape[2]):
        score = get_score(TRUE, x[:, :, k])
        all.append(score)
        print('Model %i has OOF score = %.4f' % (k, score))

    m = [np.argmin(all)]
    w = []

    old = np.min(all)

    RES = 200
    PATIENCE = args.pat
    TOL = args.tol
    DUPLICATES = False

    print('Ensemble score = %.4f by beginning with model %i' % (old, m[0]))
    print()

    for kk in range(len(OOF)):

        # BUILD CURRENT ENSEMBLE
        md = x[:, :, m[0]]
        for i, k in enumerate(m[1:]):
            md = w[i] * x[:, :, k] + (1 - w[i]) * md

        # FIND MODEL TO ADD
        mx = 1000
        mx_k = 0
        mx_w = 0
        print('Searching for best model to add... ')

        # TRY ADDING EACH MODEL
        for k in range(x.shape[2]):
            print(k, ', ', end='')
            #print(f'k= {k}, m={m}')
            if not DUPLICATES and (k in m):
                continue
            # EVALUATE ADDING MODEL K WITH WEIGHTS W
            bst_j = 0
            bst = 10000
            ct = 0
            for j in range(RES):
                #print(f'Md shape: {md.shape}')
                tmp = j / RES * x[:, :,  k] + (1 - j / RES) * md
                #print(f'tmp shape: {tmp.shape}')
                score = get_score(TRUE, tmp)
                #print(f'Score: {score}')
                if score < bst:
                    bst = score
                    bst_j = j / RES
                else:
                    ct += 1
                if ct > PATIENCE:
                    break
            #print(f'bst: {bst} mx: {mx}')
            if bst < mx:
                mx = bst
                mx_k = k
                mx_w = bst_j

        # STOP IF INCREASE IS LESS THAN TOL
        inc = old - mx
        #print(f'inc: {inc} tol: {TOL}')
        if inc <= TOL:
            print()
            print('No decrease. Stopping.')
            break

        # DISPLAY RESULTS
        print()  # print(kk,mx,mx_k,mx_w,'%.5f'%inc)
        print('Ensemble score = %.4f after adding model %i with weight %.3f. Decrease of %.4f' % (mx, mx_k, mx_w, inc))
        print()

        old = mx
        m.append(mx_k)
        w.append(mx_w)

    print(f'We are using {len(m)} models: {m}')
    print('with weights', w)
    print('and achieve a score of = %.4f' % old)

    # base model preds
    md = x[:, :,  m[0]]

    w_ = []
    for i, k in enumerate(m[1:]):
        #print(f'wt: {w[i]} {1-w[i]}')
        w_.append(1-w[i])
        md = w[i] * x[:, :, k] + (1 - w[i]) * md

    score = get_score(TRUE, md)

    """
    ensemble_df = pd.DataFrame()
    ensemble_df['text_id'] = train_df['text_id']
    ensemble_df['label'] = TRUE
    ensemble_df = pd.concat([ensemble_df, pd.DataFrame(md)], axis=1)
    ensemble_df.rename(columns={0: "Ineffective", 1:"Adequate", 2:"Effective"}, inplace=True)
    os.makedirs('./level2', exist_ok=True)
    #ensemble_df.to_csv('./level2/ensemble_df.csv', index=False)
    """

    #print('--' * 5)
    #print(f'w:{w} w_: {w_}')
    print('----'*5)

    for i, row in enumerate(m):
        print(f"'{OOF[row]}',")

    print('----'*5)
    print('-Weights-')
    wt_sum = 0
    final_wts = []
    for i, row in enumerate(m):
        if i == 0:
            wts = w_[i]
            for wt_ in w_[1:]:
                # print(wt_)
                wts *= wt_
        else:
            wts = w[i-1]
            for wt_ in w_[i:]:
                # print(wt_)
                wts *= wt_
        wts = np.round(wts, 5)
        wt_sum += wts
        print(f"{OOF[row]} wt: {wts}")
        final_wts.append(wts)
    print(f'Score: {score}')
    print(f'wts: {final_wts}')
    print(f'Wt sum: {wt_sum}')


if __name__ == "__main__":
    main()
