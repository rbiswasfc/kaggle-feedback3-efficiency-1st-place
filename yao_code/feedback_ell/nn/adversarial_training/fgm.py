"""
@created by: heyao
@created at: 2022-08-25 00:36:20
"""
import torch


class FGM(object):
    """定义对抗训练方法FGM,对模型embedding参数进行扰动"""

    def __init__(self, model, epsilon=0.25):
        """
        Usage:
        ```python
        ...
        fgm = FGM(model)
        loss.backward(
        torch.nn.utils.clip_grad_norm_(
            parameters=self.parameters(), max_norm=self.config.train.max_grad_norm
        )

        # ======== run fgm after get loss and backward ========
        fgm.attack()
        logits = model(x)
        adv_loss = self.criterion(logits, y.view(-1)).mean()
        adv_loss.backward()
        # I don't know if gradient clipping is needed.
        # torch.nn.utils.clip_grad_norm_(
        #     parameters=self.parameters(), max_norm=self.config.train.max_grad_norm
        # )
        fgm.restore()

        # ======== run the remain part of your training ========
        optimizer.step()
        ...
        ```
        :param model:
        :param epsilon:
        """
        self.model = model
        self.epsilon = epsilon
        self.backup = {}

    def attack(self, embed_name='word_embeddings'):
        """得到对抗样本

        :param embed_name:模型中embedding的参数名
        :return:
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)

                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, embed_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and embed_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
