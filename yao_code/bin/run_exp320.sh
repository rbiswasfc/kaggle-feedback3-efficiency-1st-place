#!/bin/bash

export PYTHONPATH=/home/heyao/kaggle/feedback-ells/yao_code
export PATH="/bin/bash:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_PATH=''

for i in 0 1 2 3 4 5 6 7
do
/home/heyao/miniconda3/envs/kaggle/bin/python ../feedback_ell/train/regression.py \
  --config ../config/exp320_debv3l.yaml train.fold_index="[$i]" train.save_config=true \
  dataset.path=" ..." \
  dataset.fold_file="../../input/fb3_folds/train_8folds.csv" \
  model.path="microsoft/deberta-v3-large" \
  train.save_to="../output/exp320"
done
