# feedback-ells

Feedback Prize - English Language Learning Kaggle Competition

## Dataset Requirements

Make sure Kaggle API is installed

```
cd ..
./setup.sh
```

## Create folds

```
python  ./tools/create_folds.py
```

## Train Models

### exp009a: PET

```
python ./feedback-ells/code/train_pet_epl.py \
--config-name conf_pet_epl_009a \
fold=0 \
use_wandb=false
```

Full-fit model

```
python ./feedback-ells/code/train_pet_epl.py \
--config-name conf_pet_epl_009a_full \
fold=0 \
use_wandb=false
```

### exp022b: Keywords

```
python ./feedback-ells/code/train_kw_prefix_epl.py \
--config-name conf_kw_022b_pl_8f \
fold=0 \
use_wandb=false
```

Full-fit model

```
python ./feedback-ells/code/train_kw_prefix_epl.py \
--config-name conf_kw_022b_pl_8f_all \
fold=0 \
use_wandb=false
```

### exp024: Multi-scale

```
python ./feedback-ells/code/train_multiscale.py \
--config-name conf_multiscale_024 \
fold=0 \
use_wandb=false
```

### exp024a: Multi-scale

```
python ./feedback-ells/code/train_multiscale.py \
--config-name conf_multiscale_024a \
fold=0 \
use_wandb=false
```

Full-fit model

```
python ./feedback-ells/code/train_multiscale.py \
--config-name conf_multiscale_024a_full \
fold=0 \
use_wandb=false
```

### exp026b: baseline

```
python ./feedback-ells/code/train_baseline_epl.py \
--config-name conf_baseline_026b_8f \
fold=0 \
use_wandb=false
```

Full-fit model

```
python ./feedback-ells/code/train_baseline_epl.py \
--config-name conf_baseline_026b_all \
fold=0 \
use_wandb=false
```

### exp027: SetFit

- Step 1: Training of sentence-transformer

```
python ./feedback-ells/code/train_setfit_pairwise.py \
--config-name conf_stage_1 \
fold=0 \
use_wandb=false
```

- Step 2: finetuning using trained sentence-transformer backbone

```
python ./feedback-ells/code/train_setfit_finetune.py \
--config-name conf_stage_2 \
fold=0 \
model.ckpt_path=../models/setfit/fb_model_fold_0_best.pth.tar \
use_wandb=true
```

### exp030: deberta-v3-small

```
python ./feedback-ells/code/train_small_epl.py \
--config-name conf_efficiency_small \
fold=0 \
use_wandb=false
```

Full-fit model

```
python ./feedback-ells/code/train_small_epl.py \
--config-name conf_efficiency_small_all \
fold=0 \
use_wandb=false
```

### Efficiency Model

```
python ./feedback-ells/code/train_efficiency_epl.py \
--config-name conf_efficiency_xsmall_all \
fold=0 \
use_wandb=false
```

## Synthetic Data using T5-model

Training

```
python ./feedback-ells/tools/t5_train.py \
--config_path ./feedback-ells/conf/t5/t5_training.json \
use_wandb=false
```

Inference

```
python ./feedback-ells/tools/t5_inference.py \
--config_path ./feedback-ells/conf/t5/t5_inference.json \
```
