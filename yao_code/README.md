# Description
This is the code for Feedback Prize - English Language Learning Kaggle Competition
of Yao part. The code is constructed with:
- **bin**: bash scripts to run training.
- **config**: yaml config for different models.
- **feedback_ell**: root dir.
  - **dataloaders**: helper functions for create dataloader from DataFrame
  - **input_preparer**: encoded text creator for different type of input. The final ensembles use `RegressionInputPreparer` only
  - **modules**: pytorch_lightning Model classes. This module defines model structure, optimizer, scheduler, etc. in a pytorch_lightning format. The final ensemble use `.regression.simple.FeedbackRegressionModule` only
  - **nn**: define many useful module, loss function, head, etc.
    - adversarial_training: 
    - head: different head. MLM task head, MultiSampleDropout head, ...
    - losses: loss functions
    - new_bert: no use!
    - optim: optimizers. try look_ahead here which works well in commonlit
    - poolers: different pooler. `poolers.MultiPooling` can create different pooling with pooling_name, seperated by '_'
    - robust: ema.
    - mixout_.py: mixout
    - my_cocolm.py: cocolm implementation to make CoCoLM work with official codes.
  - **train**: training code based on pytorch_lightning package.
    - full.py: full data train. used in the bash file.
    - regression.py: regular training code. used in the bash file.
  - **utils**: some helper functions
  - make_folds.py: make cross validation folds
- **inference**: bash scripts to run inference(for generating PL or oof) 

# Before training & inference
You need to set many setting to aligned with you local config.
1. PYTHONPATH: set the project path for every shell file. e.g. `PYTHONPATH=/home/heyao/kaggle/feedback-ells`
2. training config: 
   1. `dataset.path`: training set csv file
   2. `dataset.fold_file`: training set fold file
   3. `model.path`: pretrained model path
   4. `train.save_to`: a dir to save model checkpoints
3. training min-dist PL models config:
   1. above all config
   2. `train.pl.text_path`: where full_text csv, be used only in stage 1. default: /home/heyao/kaggle/feedback-ells/input/pl_data.csv 
   3. `train.pl.path`: PL labels, be used only in stage 1. default: /home/heyao/kaggle/feedback-ells/input/fold{fold}_pl.csv 
   4. `train.pl.prev_model_path`: stage1's checkpoint. be used only in stage 2. default: /home/heyao/kaggle/feedback-ells/output/exp303d_stage1/deberta-v3-large_regression_fold{fold}.ckpt 
   5. `train.pl.json_path`: min-dist text_ids. be used only in stage 1. default: /home/heyao/kaggle/feedback-ells/input/pl_ids_{fold}.json
4. training yao PL models config:
   1. above all config
   2. `train.pl.path`: PL labels. default: /home/heyao/kaggle/feedback-english-lan-learning/output/pl/pl_dbv3l_f4_fold{fold}.csv
   3. `train.pl.prev_model_path`: stage1's checkpoint. be used only in stage 2. default: /home/heyao/kaggle/feedback-ells/output/exp303_stage1/deberta-v3-large_regression_fold{fold}.ckpt

# Training
```shell
cd ./bin
# a. example of run training
chmod 777 ./run_exp300.sh
./run_exp300.sh

# b. example of run PL
./run_exp303d_stage1.sh
./run_exp303d_stage2.sh
```

# Generate PL
```shell
cd ./inference

sh predict_pl.sh
# two path may need set
# fb1_path: feedback 1st competition data path. default: /home/heyao/kaggle/feedback-effective/input/feedback-prize-2021
# fb3_path: feedback 3rd competition data path. default: /home/heyao/kaggle/feedback-english-lan-learning/input/feedback-prize-english-language-learning
# similar to predict_oof.sh
```

# Generate oof
```shell
cd ./inference

sh predict_oof.sh
# python predict_oof.py config.yaml \
#  --model_path=<model_checkpoint_path> \
#  filename="your-absolute-filename-to-save-oof.csv" debug=false \
#  pad_to_batch=true train.batch_size=2
```

# Notice
1. For exp300 exp301 exp302, use `feedback_ell.make_folds.py` to generate 5folds split file;
2. For exp303 to exp310, use team 4folds file;
3. For exp320, use team 8folds file;
4. Please remove checkpoints if you want to rerun exp;
5. 