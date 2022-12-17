import math
import os
import shutil

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.checkpoint
from sklearn.metrics import f1_score, log_loss
from torch.autograd.function import InplaceFunction
from torch.nn import Parameter
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from transformers.trainer_pt_utils import get_parameter_names
from loguru import logger
from sklearn.metrics import mean_squared_error


class RMSELoss(nn.Module):
    def __init__(self, reduction='mean', eps=1e-9):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        loss = torch.sqrt(self.mse(y_pred, y_true) + self.eps)
        if self.reduction == 'none':
            loss = loss
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss


def mcrmse_loss_fn(outputs, targets):
    #logger.debug(f'targets shape: {targets.shape} output shape: {outputs.shape}')
    colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss


def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores

dict_map = {
    0: 1.05,
    1: 1,
    2: 0.95,
    3: 0.95,
    4: 1.05,
    5: 0.95
}

def weighted_loss(y_trues, y_preds):
    idxes = y_trues.shape[1]
    loss = 0
    for k, i in enumerate(range(idxes)):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=0)
        rmse = torch.sqrt(colwise_mse)
        #print(rmse)
        loss += (rmse * dict_map[k])
        #print(loss)
        #print('---')
        #score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        #scores.append(score)
        #print(scores)
    #TARGET_COLUMNS = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']
    weighted_score = loss/6 #1/6(1.05*scores[0] + 1.0*scores[1] + 0.95*scores[2] + 0.95*scores[3] + 1.05*scores[4] + 0.95*scores[5])
    return weighted_score

def get_optimizer_grouped_parameters(model, config):
    """layerwise learning rate decay implementation
    """
    no_decay = ["bias", "LayerNorm.weight"]

    # initialize lr for task specific layer
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n or "pooler" in n],
            "weight_decay": config["weight_decay"]*1e-2,
            "lr": config["lr"],
        },
        {
            "params": [p for n, p in model.named_parameters() if (("fpe_span_attention" in n) | ("fpe_lstm_layer" in n) | ("fpe_" in n))],
            "weight_decay": config["weight_decay"],
            "lr": config["lr"],
        },
    ]

    #for i, layer in enumerate(model.base_model.named_parameters()):
    #    print(i, layer[0])

    # initialize lrs for every layer
    if 'bart' in config["base_model_path"]:
        layers = list(model.base_model.encoder.layers)
    elif 'funnel' in config["base_model_path"]:
            layers = [model.base_model.embeddings] + list(model.base_model.encoder.blocks)
    else:
        layers = [model.base_model.embeddings] + list(model.base_model.encoder.layer)

    layers.reverse()
    lr = config["lr"]

    for layer in layers:
        lr *= config["llrd"]
        optimizer_grouped_parameters += [
            {
                "params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": config["weight_decay"],
                "lr": lr,
            },
            {
                "params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            },
        ]
    return optimizer_grouped_parameters


def get_optimizer(model, config):
    """optimizer for model training
    """

    if 'bart' in config["base_model_path"]:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, config)
    else:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, config)

    optimizer_kwargs = {
        "betas": (config["beta1"], config["beta2"]),
        "eps": config['eps'],
    }

    optimizer_kwargs["lr"] = config["lr"]

    if config["use_bnb"]:
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(config['beta1'], config['beta2']),
            eps=config['eps'],
            lr=config['lr'],
        )
    else:
        if 'funnel' in config["base_model_path"]:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=config["lr"],
                eps=config['eps']
            )
        else:
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=config["lr"],
                eps=config['eps']
            )

    return optimizer


def get_scheduler(optimizer, warmup_steps, total_steps):
    """scheduler for model
    """
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    return scheduler


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
    os.makedirs(config["model_dir"], exist_ok=True)
    name = f"fpe_model_fold_{config['fold']}"
    filename = f'{config["model_dir"]}/{name}.pth.tar'
    torch.save(state, filename, _use_new_zipfile_serialization=False)
    if is_best:
        shutil.copyfile(filename, f'{config["model_dir"]}/{name}_best.pth.tar')

def save_checkpoint_beta(config, state, is_best):
    os.makedirs(config["model_dir"], exist_ok=True)
    name = f"fpe_model_fold_{config['fold']}"

    if is_best:
        filename = f'{config["model_dir"]}/{name}_best.pth.tar'
        torch.save(state, filename, _use_new_zipfile_serialization=False)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Created by: Cheolhyoung Lee
# Department of Mathematical Sciences, KAIST
## Email: cheolhyoung.lee@kaist.ac.kr
# Implementation of mixout from https://arxiv.org/abs/1909.11299
## "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class Mixout(InplaceFunction):
    # target: a weight tensor mixes with a input tensor
    # A forward method returns
    # [(1 - Bernoulli(1 - p) mask) * target + (Bernoulli(1 - p) mask) * input - p * target]/(1 - p)
    # where p is a mix probability of mixout.
    # A backward returns the gradient of the forward method.
    # Dropout is equivalent to the case of target=None.
    # I modified the code of dropout in PyTorch.
    @staticmethod
    def _make_noise(input):
        return input.new().resize_as_(input)

    @classmethod
    def forward(cls, ctx, input, target=None, p=0.0, training=False, inplace=False):
        if p < 0 or p > 1:
            raise ValueError("A mix probability of mixout has to be between 0 and 1," " but got {}".format(p))
        if target is not None and input.size() != target.size():
            raise ValueError(
                "A target tensor size must match with a input tensor size {},"
                " but got {}".format(input.size(), target.size())
            )
        ctx.p = p
        ctx.training = training

        if ctx.p == 0 or not ctx.training:
            return input

        if target is None:
            target = cls._make_noise(input)
            target.fill_(0)
        target = target.to(input.device)

        if inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        ctx.noise = cls._make_noise(input)
        if len(ctx.noise.size()) == 1:
            ctx.noise.bernoulli_(1 - ctx.p)
        else:
            ctx.noise[0].bernoulli_(1 - ctx.p)
            ctx.noise = ctx.noise[0].repeat(input.size()[0], 1)
        ctx.noise.expand_as(input)

        if ctx.p == 1:
            output = target
        else:
            output = ((1 - ctx.noise) * target + ctx.noise * output - ctx.p * target) / (1 - ctx.p)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.p > 0 and ctx.training:
            return grad_output * ctx.noise, None, None, None, None
        else:
            return grad_output, None, None, None, None


def mixout(input, target=None, p=0.0, training=False, inplace=False):
    return Mixout.apply(input, target, p, training, inplace)


class MixLinear(torch.nn.Module):
    __constants__ = ["bias", "in_features", "out_features"]
    # If target is None, nn.Sequential(nn.Linear(m, n), MixLinear(m', n', p))
    # is equivalent to nn.Sequential(nn.Linear(m, n), nn.Dropout(p), nn.Linear(m', n')).
    # If you want to change a dropout layer to a mixout layer,
    # you should replace nn.Linear right after nn.Dropout(p) with Mixout(p)

    def __init__(self, in_features, out_features, bias=True, target=None, p=0.0):
        super(MixLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()
        self.target = target
        self.p = p

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return F.linear(input, mixout(self.weight, self.target, self.p, self.training), self.bias)

    def extra_repr(self):
        type = "drop" if self.target is None else "mix"
        return "{}={}, in_features={}, out_features={}, bias={}".format(
            type + "out", self.p, self.in_features, self.out_features, self.bias is not None
        )


def apply_mixout(model, p):
    for sup_module in model.modules():
        for name, module in sup_module.named_children():
            if isinstance(module, nn.Dropout):
                module.p = 0.0
            if isinstance(module, nn.Linear):
                target_state_dict = module.state_dict()
                bias = True if module.bias is not None else False
                new_module = MixLinear(
                    module.in_features, module.out_features, bias, target_state_dict["weight"], p
                )
                new_module.load_state_dict(target_state_dict)
                setattr(sup_module, name, new_module)
    return model
