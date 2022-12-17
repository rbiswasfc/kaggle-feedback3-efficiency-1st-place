import os
import random
import shutil

import numpy as np
import torch
import wandb
from pynvml import (nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo,
                    nvmlInit)
from sklearn.metrics import mean_squared_error

LABEL_COLS = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def init_wandb(config):
    entity = config["wandb"]["entity"]
    project = config["wandb"]["project"]
    tags = config["tags"]
    if config["all_data"]:
        run_id = f"{config['wandb']['run_name']}-all-data"
    else:
        run_id = f"{config['wandb']['run_name']}-fold-{config['fold']}"

    run = wandb.init(
        entity=entity,
        project=project,
        config=config,
        tags=tags,
        name=run_id,
        anonymous="must",
        job_type="Train",
    )

    return run


def print_gpu_utilization():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']*1e6


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(config, state, is_best):
    os.makedirs(config["outputs"]["model_dir"], exist_ok=True)
    name = f"fb_model_fold_{config['fold']}"

    filename = f'{config["outputs"]["model_dir"]}/{name}.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)

    if is_best:
        shutil.copyfile(filename, f'{config["outputs"]["model_dir"]}/{name}_best.pth.tar')


def get_score(y_trues, y_preds):
    scores = dict()
    num_targets = y_trues.shape[1]
    assert num_targets == len(LABEL_COLS), "target size mismatch"

    for i in range(num_targets):
        target_name = LABEL_COLS[i]
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]
        score = mean_squared_error(y_true, y_pred, squared=False)
        scores[target_name] = score

    scores["lb"] = np.mean([v for k, v in scores.items()])

    for k, v in scores.items():
        scores[k] = round(v, 5)
    return scores
