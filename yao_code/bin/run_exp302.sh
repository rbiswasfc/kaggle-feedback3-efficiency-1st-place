#!/bin/bash

export PYTHONPATH=/home/heyao/kaggle/feedback-ells/yao_code
export PATH="/bin/bash:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_PATH=''

for i in 0 1 2 3 4
do
/home/heyao/miniconda3/envs/kaggle/bin/python ../feedback_ell/train/regression.py \
  --config ../config/exp302_debl.yaml train.fold_index="[$i]" train.save_config=true \
  train.save_to="../output/exp302" \
  dataset.path=" ..." \
  dataset.fold_file="../../input/fb3_folds.csv" \
  model.path="microsoft/deberta-large"
done
