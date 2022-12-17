"""
@created by: heyao
@created at: 2022-09-22 03:31:32
"""
import torch
from omegaconf import DictConfig

from feedback_ell.modules.base import BaseLightningModule
from feedback_ell.nn import EMA


class EMARegressionModule(BaseLightningModule):
    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.automatic_optimization = False
        self.awp = None
        self.start = False

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
        # start ema step
        if self.best_score < self.config.train.ema.get("from_score", 100):
            if not self.start:
                self.ema.register()
            self.start = True
            self.ema.update()
        scheduler = self.lr_schedulers()
        scheduler.step()

        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy(), logits.detach().cpu().numpy())
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def on_train_start(self) -> None:
        if self.config.train.ema.enable:
            # epoch start with 0
            self.ema = EMA(self, decay=self.config.train.ema.decay)
            self.ema.register()
            self.automatic_optimization = False

    def on_validation_start(self):
        super(EMARegressionModule, self).on_validation_start()
        if self.ema is not None and self.best_score < self.config.train.ema.get("from_score", 100):
            self.apply_shadow = True
            self.ema.apply_shadow()

    def on_validation_end(self) -> None:
        super(EMARegressionModule, self).on_validation_end()
        if self.ema is not None and self.best_score < self.config.train.ema.get("from_score", 100) and self.apply_shadow:
            self.ema.restore()
        self.apply_shadow = False
