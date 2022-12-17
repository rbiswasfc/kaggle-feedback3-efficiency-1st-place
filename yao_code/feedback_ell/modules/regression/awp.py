"""
@created by: heyao
@created at: 2022-09-08 12:28:35
"""
import torch
from omegaconf import DictConfig

from feedback_ell.modules.base import BaseLightningModule
from feedback_ell.modules.regression.label_inject import LabelInjectWithAvgRegressionModule
from feedback_ell.nn import AWP


class AWPMixIn(BaseLightningModule):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.automatic_optimization = False
        self.awp = None

    def training_step(self, batch, batch_index):
        x, y = batch
        optimizer = self.optimizers(use_pl_optimizer=True)
        logits = self(batch)
        loss = self.compute_loss(logits, y)

        optimizer.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(
            parameters=self.parameters(), max_norm=self.config.train.max_grad_norm
        )
        optimizer.step()

        # start awp step
        if self.best_score < self.config.train.awp.from_score:
            adv_loss = self.awp.attack_backward(x, y, epoch=self.current_epoch)
            if adv_loss is not None:
                self.manual_backward(adv_loss)
                # torch.nn.utils.clip_grad_norm_(
                #     parameters=self.parameters(), max_norm=self.config.train.max_grad_norm
                # )
                self.awp._restore()
                optimizer.step()
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_start(self) -> None:
        if self.config.train.awp.enable:
            self.awp = AWP(self, self.optimizers(use_pl_optimizer=True), adv_param=self.config.train.awp.adv_param,
                           adv_lr=self.config.train.awp.adv_lr, adv_eps=self.config.train.awp.adv_eps,
                           start_epoch=self.config.train.awp.get("start_epoch", 1),
                           adv_step=self.config.train.awp.get("adv_step", 1))
            self.automatic_optimization = False

class AWPRegressionModule(AWPMixIn, BaseLightningModule):
    pass


class AWPLabelInjectRegressionModule(AWPMixIn, LabelInjectWithAvgRegressionModule):
    pass
