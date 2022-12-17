"""
@created by: heyao
@created at: 2022-08-25 00:15:53
"""
import gc
import math
import time
from functools import partial

from feedback_ell.nn.heads.msd import MultiSampleDropout
from feedback_ell.nn.heads.separate_attention_head import SeparatedHead, ResidualSeparatedHead
from feedback_ell.nn.new_bert.deberta_modeling import DebertaEncoder
from feedback_ell.nn.optim.look_ahead import Lookahead
from feedback_ell.utils.meter import AverageMeter, CumsumMeter
from feedback_ell.utils.metrics import competition_score

try:
    import bitsandbytes as bnb
except ImportError:
    bnb = None
import torch
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoModel, get_scheduler, T5EncoderModel, get_cosine_with_hard_restarts_schedule_with_warmup
from transformers import DebertaModel

from feedback_ell.nn import losses
from feedback_ell.nn.poolers import MultiPooling
from feedback_ell.utils.stable_training import differential_learning_rate, get_optimizer_params, reinit_last_layers

import math

from torch.optim.lr_scheduler import LambdaLR


# def get_cosine_with_hard_restarts_schedule_with_warmup(
#         optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 2, restart_ratio: float = 1.0,
#         last_epoch: int = -1
# ):
#     """
#     Create a schedule with a learning rate that decreases following the values of the cosine function between the
#     initial lr set in the optimizer to 0, with several hard restarts, after a warmup period during which it increases
#     linearly between 0 and the initial lr set in the optimizer.
#     Args:
#         optimizer ([`~torch.optim.Optimizer`]):
#             The optimizer for which to schedule the learning rate.
#         num_warmup_steps (`int`):
#             The number of steps for the warmup phase.
#         num_training_steps (`int`):
#             The total number of training steps.
#         num_cycles (`int`, *optional*, defaults to 1):
#             The number of hard restarts to use.
#         last_epoch (`int`, *optional*, defaults to -1):
#             The index of the last epoch when resuming training.
#     Return:
#         `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
#     """
#     num_cycles = 2
#
#     def lr_lambda(current_step):
#         if current_step < num_warmup_steps:
#             return float(current_step) / float(max(1, num_warmup_steps))
#         progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
#         if progress >= 1.0:
#             return 0.0
#         if progress < 0.5:
#             return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))
#         return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))) * restart_ratio
#
#     return LambdaLR(optimizer, lr_lambda, last_epoch)


class BaseLightningModule(pl.LightningModule):
    """base model for lightning

    This is model module for regression with 6 labels and mcrmse loss.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        # initial layers
        self.backbone = self.init_backbone(config.model.path)
        self.customer_pooling = MultiPooling(
            config.model.pooling,
            hidden_size=self.backbone.config.hidden_size,
            num_hidden_layers=self.backbone.config.num_hidden_layers,
            layer_start=self.config.model.get("layer_start", 4)
        )
        feature_size = self.get_hidden_size(config.model.pooling, self.backbone.config.hidden_size)
        self.feature_size = feature_size
        self._create_head(feature_size)

        if config.train.loss == "mse":
            self.criterion = nn.MSELoss()
        elif config.train.loss == "rmse":
            self.criterion = nn.MSELoss()
        elif config.train.loss == "mcrmse":
            self.criterion = losses.mcrmse
        elif config.train.loss == "smoothl1":
            self.criterion = nn.SmoothL1Loss()
        elif config.train.loss == "mcrmse_smoothl1":
            l1 = losses.mcrmse
            l2 = nn.SmoothL1Loss()
            self.criterion = lambda x, y: l1(x, y) + l2(x, y)
        elif config.train.loss == "weighted_mse":
            weights = None
            self.criterion = partial(losses.weighted.weighted_mse_loss, weights=weights)
        elif config.train.loss == "weighted_focal_mse":
            # self.criterion = partial(losses.balance_mse.bmc_loss_md, noise_var=1.0)
            # self.criterion = losses.balance_mse.MultiTargetBMCLoss(init_noise_sigma=0.5)
            self.criterion = partial(losses.weighted.weighted_focal_mse_loss, activate=config.train.loss_activation)
        elif config.train.loss == "huber":
            self.criterion = losses.weighted.weighted_huber_loss
        else:
            print(f"maybe invalid loss: {config.train.loss}")

        # metrics and criterion
        self.train_losses = AverageMeter()
        self.val_losses = AverageMeter()
        self.train_metrics = CumsumMeter(competition_score)
        self.val_metrics = CumsumMeter(competition_score)

        # tricks
        reinit_last_layers(self.backbone, num_layers=config.model.num_reinit_layers)
        if config.train.gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()
            self.backbone.config.use_cache = False
        self.best_score = 10

    def _create_head(self, feature_size):
        model_head = self.config.model.get("head", "")
        if self.config.model.msd:
            self.head = MultiSampleDropout(feature_size, 6)
        elif "sep_" in model_head:
            self.head = ResidualSeparatedHead(feature_size, 6, pooling=model_head.split("_", 1)[-1])
        else:
            self.head = nn.Linear(feature_size, 6)

    def init_backbone(self, model_path):
        if "t5" in model_path:
            model_class = T5EncoderModel
        else:
            model_class = AutoModel
        if self.config.train.zero_dropout:
            if "bart" in self.config.model.path:
                kwargs = {"attention_dropout": 0, "dropout": 0, "activation_dropout": 0}
            else:
                kwargs = {"hidden_dropout_prob": 0, "attention_probs_dropout_prob": 0}
        else:
            kwargs = {}
        if "roberta" in self.config.model.path:
            kwargs["add_pooling_layer"] = False
        # kwargs["position_biased_input"] = True
        backend = model_class.from_pretrained(model_path, output_hidden_states=True, **kwargs)
        if self.config.model.residual_transformers:
            if "deberta" in self.config.model.path and "deberta-v3" not in self.config.model.path:
                backend.encoder = DebertaEncoder(backend.config)
        return backend

    def get_hidden_size(self, pooling_name, hidden_size):
        if any(i in pooling_name for i in ["meanmax", "attnmean"]):
            return hidden_size * 2
        return hidden_size

    def get_feature(self, input_ids, attention_mask=None, **kwargs):
        model_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        if hasattr(model_output, "hidden_states"):
            hidden_states = model_output.hidden_states
        else:
            hidden_states = [model_output.last_hidden_state]
        if "weighted" in self.config.model.pooling or "residual" in self.config.model.pooling:
            pooler_output = self.customer_pooling(hidden_states, attention_mask)
        else:
            pooler_output = self.customer_pooling(hidden_states[-1], attention_mask)
        return pooler_output

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        feature = self.get_feature(x["input_ids"], x["attention_mask"])
        out = self.head(feature)
        return out

    def compute_loss(self, logits, labels):
        return self.criterion(logits, labels)

    def configure_scheduler(self, optimizer):
        batch_size = self.config.train.batch_size
        num_training_steps_per_epoch = self.config.dataset.n_samples_in_train / batch_size
        num_training_steps = int(math.ceil(num_training_steps_per_epoch) * self.config.train.epochs)
        warmup_steps = self.config.optim.scheduler.num_warmup_steps
        warmup_ratio = self.config.optim.scheduler.get("warmup_ratio", 0)
        if warmup_ratio > 0:
            warmup_steps = int(warmup_ratio * num_training_steps)
        if self.config.optim.scheduler.name in ["linear", "cosine"]:
            scheduler = get_scheduler(self.config.optim.scheduler.name, optimizer,
                                      num_warmup_steps=warmup_steps,
                                      num_training_steps=num_training_steps, **self.config.optim.scheduler.kwargs)
        elif self.config.optim.scheduler.name == "cosine_restart":
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
                num_cycles=self.config.optim.scheduler.get("num_cycles", 3),
                # restart_ratio=self.config.optim.scheduler.get("restart_ratio", 1.0),
            )
        else:
            raise ValueError(f"no scheduler named {self.config.optim.scheduler.name}")
        return scheduler

    def configure_optimizers(self):
        if self.config.model.differential_lr.enable:
            print(f"enable differential learning rate: {self.config.optim.optimizer.lr}")
            model_params = differential_learning_rate(self, encoder_lr=self.config.optim.optimizer.lr,
                                                      decoder_lr=self.config.optim.optimizer.head_lr,
                                                      weight_decay=self.config.optim.optimizer.weight_decay,
                                                      lr_factor=self.config.model.differential_lr.lr_factor)
        else:
            model_params = get_optimizer_params(self,
                                                encoder_lr=self.config.optim.optimizer.lr,
                                                decoder_lr=self.config.optim.optimizer.head_lr,
                                                weight_decay=self.config.optim.optimizer.weight_decay)
        if bnb is None:
            print("bitsandbytes not install, use initial optimizer")
            base_optimizer = torch.optim.AdamW(
                model_params,
                lr=self.config.optim.optimizer.head_lr,
                weight_decay=self.config.optim.optimizer.weight_decay,
                eps=self.config.optim.optimizer.eps,
                betas=self.config.optim.optimizer.betas
            )
        else:
            base_optimizer = bnb.optim.AdamW8bit(
                model_params,
                lr=self.config.optim.optimizer.head_lr,
                weight_decay=self.config.optim.optimizer.weight_decay,
                eps=self.config.optim.optimizer.eps,
                betas=self.config.optim.optimizer.betas
            )
        if self.config.optim.optimizer.get("lookahead", False):
            optimizer = Lookahead(base_optimizer, k=5, alpha=0.5)
            optimizer.defaults = base_optimizer.defaults
        else:
            optimizer = base_optimizer
        scheduler = self.configure_scheduler(optimizer)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        # outputs are outputs of corresponding xxxing_step
        self.train_metrics.reset()
        self.train_losses.reset()
        gc.collect()
        self.t0 = time.perf_counter()

    def validation_step(self, batch, batch_index):
        x, y = batch
        logits = self(batch)
        loss = self.compute_loss(logits, y)
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_start(self):
        self.val_metrics.reset()
        self.val_losses.reset()
        gc.collect()
        self.val_t0 = time.perf_counter()

    def validation_epoch_end(self, outputs):
        super(BaseLightningModule, self).validation_epoch_end(outputs)
        self.log("val/score", self.val_metrics.score, prog_bar=False, on_step=False, on_epoch=True)
        gc.collect()
        score = self.val_metrics.score
        self.best_score = min([self.best_score, score])
        print(f"[{time.perf_counter() - self.val_t0:.2f}] best score {self.best_score}({score})")

    def on_train_epoch_end(self) -> None:
        super(BaseLightningModule, self).on_train_epoch_end()
        print(f"train use: {time.perf_counter() - self.t0:.2f} seconds")
        print("-" * 60)


if __name__ == '__main__':
    from feedback_ell.modules.regression.awp import AWPRegressionModule, AWPLabelInjectRegressionModule
    from feedback_ell.modules.regression.label_inject import LabelInjectRegressionModule, \
        LabelInjectWithAvgRegressionModule
    from feedback_ell.modules.regression.simple import RegressionWithMLMAuxiliaryTask

    config = OmegaConf.load("../../config/deberta_v3_large_reg.yaml")
    # config.model.pooling = "lstm"
    print(config)
    model = RegressionWithMLMAuxiliaryTask(config)
    # model.backbone.resize_token_embeddings(128003)
    # [[31066], [16843], [11174], [5741, 8495], [10877], [15521]]
    # input_ids = torch.LongTensor([[1, 12, 23, 34, 128001, 128002, 2]])
    input_ids = torch.LongTensor([[1, 31066, 12, 312, 123, 52321, 1231, 1231, 54, 123, 1231, 12312, 412, 123, 3, 4, 2]])
    attention_mask = torch.LongTensor([[1] * input_ids.shape[1]])
    labels = torch.FloatTensor([[1.5, 2, 2, 3, 2, 1]])
    mask_input_ids = torch.LongTensor(
        [[1, 31066, 128000, 312, 123, 128000, 1231, 1231, 54, 123, 1231, 128000, 412, 123, 3, 4, 2]])
    mask_labels = torch.LongTensor(
        [[-100, -100, 12, -100, -100, 52321, -100, -100, -100, -100, -100, 12312, -100, -100, -100, -100, -100]])
    model.training_step([{
        "input_ids": input_ids, "attention_mask": attention_mask,
        "mask_input_ids": mask_input_ids, "mask_labels": mask_labels
    }, labels], 0)
    # logits = model([{"input_ids": input_ids, "attention_mask": attention_mask}, 1])
    # print(f"{logits = }")
    # print(f"{logits.shape = }")
    # print(f"{model.compute_loss(logits, labels) = }")
