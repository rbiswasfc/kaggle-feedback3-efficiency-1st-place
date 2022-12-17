#!/bin/bash

export PYTHONPATH=/home/heyao/kaggle/feedback-ells/yao_code
export PATH="/bin/bash:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_PATH=''

for i in 0 1 2 3
do
/home/heyao/miniconda3/envs/kaggle/bin/python ../feedback_ell/train/regression.py \
  --config ../config/exp310_debv3l.yaml train.fold_index="[$i]" train.save_config=true \
  train.pl.save_last_checkpoint=false optim.optimizer.lr=2e-5 optim.optimizer.head_lr=1e-3 \
  dataset.path=" ..." \
  dataset.fold_file="../../input/fb3_folds/train_4folds.csv" \
  model.path="microsoft/deberta-v3-large" \
  train.pl.stage=1 train.save_to="../output/exp310_stage1" \
  train.pl.path="/home/heyao/kaggle/feedback-english-lan-learning/output/pl/pl_dbv3l_f4_fold{fold}.csv"
done

# /home/heyao/kaggle/feedback-english-lan-learning/output/pl/pl_dbv3l_f4_fold{fold}.csv is
# generated using exp303 only
