#!/bin/bash

export PYTHONPATH=/home/heyao/kaggle/feedback-ells/yao_code
export PATH="/bin/bash:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_PATH=''

/home/heyao/miniconda3/envs/kaggle/bin/python ../feedback_ell/train/full.py \
  --config ../config/exp303_debv3l.yaml train.save_config=true \
  train.save_to="../output/exp303_full" \
  dataset.path=" ..." \
  dataset.fold_file="../../input/fb3_folds/train_4folds.csv" \
  model.path="microsoft/deberta-v3-large"
