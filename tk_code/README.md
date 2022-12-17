# feedback-ells

Feedback Prize - English Language Learning Kaggle Competition

## Dataset Requirements

Make sure Kaggle API is installed

```
cd ..
./setup.sh
```


```
cd code
```

## Train Models


### exp203b: deberta-large 4 fold model

```
python train_fts.py --config configs/exp_config_del_4fold.json --use_wandb --fold 0/1/2/3
```

### exp208: deberta-large 8 fold model

```
python train_fts.py --config configs/exp_config_del.json --use_wandb --fold 0/1/2/3
```

### exp207a: deberta-v3-large 8 fold PET model
```
python train_pet.py --config-name debv3_large_prompt fold=1 use_wandb=true outputs.model_dir=../output_new/exp207a-debv3l-new2 wandb.run_name=exp207a-debv3l-new2 fold=0/1/2/3/4/5/6/7
```

### exp209f: deberta-v3-base Full Data model

```
python train_fts.py --config configs/exp_config_dbase.json --use_wandb --fold 0
```