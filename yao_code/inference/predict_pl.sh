#!/bin/bash

export PYTHONPATH=/home/heyao/kaggle/feedback-ells
export PATH="/bin/bash:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_PATH=''

/home/heyao/miniconda3/envs/kaggle/bin/python predict_pl.py \
  /home/heyao/kaggle/feedback-ells/output/exp303_stage2/exp303_debv3l.yaml \
  --model_path=/home/heyao/kaggle/feedback-ells/output/exp303_stage2 \
  filename="/home/heyao/kaggle/feedback-ells/output/pl/pl_team4fold_303only_fold{fold}.csv" debug=false \
  fb1_path=null \
  fb3_path=null \
  train.batch_size=16
# fb1_path: feedback 1st competition data path. default: /home/heyao/kaggle/feedback-effective/input/feedback-prize-2021
# fb3_path: feedback 3rd competition data path. default: /home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning
