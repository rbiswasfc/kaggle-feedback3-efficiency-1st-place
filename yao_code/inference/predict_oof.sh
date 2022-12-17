#!/bin/bash
export PYTHONPATH=/home/heyao/kaggle/feedback-ells
export PATH="/bin/bash:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/usr/local/cuda-11.4/bin"
export LD_LIBRARY_PATH=''
export CUDA_HOME=''
export CUDA_PATH=''

/home/heyao/miniconda3/envs/kaggle/bin/python predict_oof.py \
  /home/heyao/kaggle/feedback-ells/output/exp303d_stage2/exp303d_debv3l.yaml \
  --model_path=/home/heyao/kaggle/feedback-ells/output/exp303d_stage2 \
  filename="/home/heyao/kaggle/feedback-english-lan-learning/output/oof/exp303d_stage2.csv" debug=false \
  model.path="/media/heyao/42f00068-ba8d-48dd-8719-61eda5244b8d/pretrained-models/deberta-v3-large" \
  pad_to_batch=true train.batch_size=2
