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
  train.pl.save_last_checkpoint=false optim.optimizer.lr=7e-6 optim.optimizer.head_lr=1e-5 \
  dataset.path=" ..." \
  dataset.fold_file="../../input/fb3_folds/train_4folds.csv" \
  model.path="microsoft/deberta-v3-large" \
  train.pl.stage=2 train.save_to="../output/exp310_stage2" \
  train.pl.prev_model_path="../output/exp310_stage1/deberta-v3-large_regression_fold{fold}.ckpt"
done
