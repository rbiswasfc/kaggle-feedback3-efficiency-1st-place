#!/bin/bash

export PYTHONPATH=/home/heyao/kaggle/feedback-ells/yao_code
export PATH="/bin/bash:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_PATH=''

for i in 0 1 2 3
do
/home/heyao/miniconda3/envs/kaggle/bin/python ../feedback_ell/train/regression.py \
  --config ../config/exp303d_debv3l.yaml train.fold_index="[$i]" train.save_config=true \
  train.pl.save_last_checkpoint=true optim.optimizer.lr=1e-5 optim.optimizer.head_lr=2e-4 \
  train.validation_interval=0.5 train.epochs=4 train.pl.stage=1 \
  dataset.path=" ..." \
  dataset.fold_file="../../input/fb3_folds/train_4folds.csv" \
  model.path="microsoft/deberta-v3-large" \
  train.save_to="../output/exp303d_stage1" \
  train.pl.path=/home/heyao/kaggle/feedback-ells/input/fold{fold}_pl.csv
  train.pl.json_path=/home/heyao/kaggle/feedback-ells/input/pl_ids_{fold}.json
  train.pl.text_path="/home/heyao/kaggle/feedback-ells/input/pl_data.csv"
done
# min-dist PL have 4 paths
# text_path: where full_text csv, be used only in stage 1. default: /home/heyao/kaggle/feedback-ells/input/pl_data.csv
# path: PL labels, be used only in stage 1. default: /home/heyao/kaggle/feedback-ells/input/fold{fold}_pl.csv
# prev_model_path: stage1's checkpoint. be used only in stage 2. default: /home/heyao/kaggle/feedback-ells/output/exp303d_stage1/deberta-v3-large_regression_fold{fold}.ckpt
# json_path: min-dist text_ids. be used only in stage 1. default: /home/heyao/kaggle/feedback-ells/input/pl_ids_{fold}.json
