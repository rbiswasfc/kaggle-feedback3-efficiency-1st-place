import pdb
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel, BertConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.deberta_v2.modeling_deberta_v2 import \
    DebertaV2OnlyMLMHead

try:
    file = Path(__file__).resolve()
    parent, root = file.parent, file.parents[1]
    sys.path.insert(0, str(root))
    from rb_utils.head_utils import (BertAttentionFeedbackHead,
                                  DepthLstmFeedbackHead,
                                  DepthResidualFeedbackHead,
                                  LstmBertAttentionFeedbackHead, MeanPooling,
                                  TargetTokenFeedbackHead)
except Exception as e:
    print(e)
    raise ImportError

from loguru import logger
#-------- Loss Functions -------------------------------------------------#

def mcrmse_loss_fn(outputs, targets):
    colwise_mse = torch.mean(torch.square(targets - outputs), dim=0)
    loss = torch.mean(torch.sqrt(colwise_mse), dim=0)
    return loss

class RMSELoss(nn.Module):
    """
    credit: https://www.kaggle.com/code/yasufuminakama/fb3-deberta-v3-base-baseline-train
    """

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


class MCRMSELoss(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction == "none":
            raise NotImplementedError
        self.rmse = RMSELoss(reduction=reduction)

    def forward(self, preds, labels):
        num_targets = labels.shape[1]
        score = 0
        for i in range(num_targets):
            score += self.rmse(preds[:, i], labels[:, i]) / num_targets
        return score

#-------- AWP ------------------------------------------------------------#


class AWP:
    """Implements weighted adverserial perturbation
    adapted from: https://www.kaggle.com/code/wht1996/feedback-nn-train/notebook
    """

    def __init__(self, model, optimizer, adv_param="weight", adv_lr=1, adv_eps=0.0001):
        self.model = model
        self.optimizer = optimizer
        self.adv_param = adv_param
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}
        self.backup_eps = {}

    def attack_backward(self, batch, accelerator):
        if self.adv_lr == 0:
            return
        self._save()
        self._attack_step()

        _, adv_loss, _ = self.model(**batch)
        self.optimizer.zero_grad()
        accelerator.backward(adv_loss)
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self.backup_eps[name][0]), self.backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and self.adv_param in name:
                if name not in self.backup:
                    self.backup[name] = param.data.clone()
                    grad_eps = self.adv_eps * param.abs().detach()
                    self.backup_eps[name] = (
                        self.backup[name] - grad_eps,
                        self.backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}
        self.backup_eps = {}

#-------- Re-initialization ------------------------------------------------------#


def reinit_deberta(backbone, num_reinit_layers):
    """
    reinitialize top `num_reinit_layers` of the backbone
    """
    config = backbone.config

    for layer in backbone.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Prompt Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class FeedbackModelPrompt(nn.Module):
    """
    The feedback-ells model for prompt based fine-tuning
    """

    def __init__(self, config):
        print("initializing the feedback model...")

        super(FeedbackModelPrompt, self).__init__()
        config = config["model"]
        self.config = config
        self.target_names = config["target_names"]
        self.num_targets = len(self.target_names)

        self.lb = 1.0  # lowest score
        self.ub = 5.0  # highest score

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["backbone_path"])

        #----------------------------- Backbone -----------------------------------------#
        backbone_config = AutoConfig.from_pretrained(self.config["backbone_path"])
        backbone_config.update(
            {
                "hidden_dropout_prob": 0.0,
                "attention_probs_dropout_prob": 0.0,
            }
        )

        self.backbone = AutoModel.from_pretrained(self.config["backbone_path"], config=backbone_config)

        # resize model embeddings
        print("resizing model embeddings...")
        print(f"tokenizer length = {config['len_tokenizer']}")
        self.backbone.resize_token_embeddings(config["len_tokenizer"])

        # enable gradient checkpointing
        self.backbone.gradient_checkpointing_enable()

        # re-initialization
        if config["num_layers_reinit"] > 0:
            print(f"re-initializing last {self.config['num_layers_reinit']} layers of the base model...")
            reinit_deberta(self.backbone, self.config["num_layers_reinit"])

        # freeze embeddings
        if config["n_freeze"] > 0:
            print(f"setting requires grad to false for last {config['n_freeze']} layers")
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.layer[:config["n_freeze"]].requires_grad_(False)

        #----------------------------- Head --------------------------------------------#
        # self.classifier = DebertaV2OnlyMLMHead(backbone_config)

        self.classifiers = nn.ModuleList([DebertaV2OnlyMLMHead(backbone_config) for _ in range(self.num_targets)])

        #----------------------------- Loss --------------------------------------------#
        print(f">>> Model training will use BCEWithLogitsLoss loss function")
        if self.config["loss_fn"] == 'bce':
            self.loss_fn = nn.BCELoss(reduction='mean')
        elif self.config["loss_fn"] == 'smooth_l1':
            self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1)


    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        target_token_idxs=None,
        labels=None,
        pos_tok_id=None,
        neg_tok_id=None,
        **kwargs
    ):
        bs = input_ids.shape[0]  # batch size

        backbone_outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        sequence_output = backbone_outputs[0]

        # pdb.set_trace()

        target_token_idxs = target_token_idxs[0]  # same for all examples in the batch
        sequence_output = sequence_output[:, target_token_idxs]  # (bs, 6, h)

        # prediction_scores = self.classifier(sequence_output)  # (bs, 6, num_vocab)
        #print(f'seq output: {sequence_output}')

        prediction_scores = [classifier(sequence_output[:, target_idx])
                             for target_idx, classifier in enumerate(self.classifiers)]  # [(bs, num_vocab), (bs, num_vocal)...]
        prediction_scores = torch.stack(prediction_scores, dim=1)  # (bs, 6, num_vocab)

        #print(f'prediction_scores: {prediction_scores}')
        #print(f'pos tok id: {pos_tok_id}\n neg tok id: {neg_tok_id}')

        # Check top predictions for debug
        """
        for i in range(bs):
            for j in range(self.num_targets):
                top_2_tokens = torch.argsort(prediction_scores[i, j, :], descending=True)[:3].tolist()
                print(f'Top tokens:')
                for token in top_2_tokens:
                    answer = self.tokenizer.decode([token]).replace(' ','')
                    print(f"Prediction: {answer}")
        """

        if isinstance(pos_tok_id, list):
            pos_scores = torch.mean(prediction_scores[:, :, pos_tok_id], dim=-1).unsqueeze(-1)  # (bs, 6, 1)
        else:
            pos_scores = prediction_scores[:, :, pos_tok_id].unsqueeze(-1)  # (bs, 6, 1)

        if isinstance(neg_tok_id, list):
            neg_scores = torch.mean(prediction_scores[:, :, neg_tok_id], dim=-1).unsqueeze(-1)  # (bs, 6, 1)
        else:
            neg_scores = prediction_scores[:, :, neg_tok_id].unsqueeze(-1)  # (bs, 6, 1)

        #print(f'pos scores: {pos_scores} neg scores: {neg_scores}')

        # pdb.set_trace()

        if self.config["only_pos_tok"]:
            logits = pos_scores.squeeze()
            orig_logits =  torch.cat([pos_scores, neg_scores], dim=-1)
        else:
            logits = torch.cat([pos_scores, neg_scores], dim=-1)  # (bs, 6, 2)

        #print(f'logits shape: {logits.shape}')
        if self.config["loss_fn"] == 'bce':
            logits = F.softmax(logits, dim=-1)  # (bs, 6, 2)

        #print(f'logits after softmax: {logits}')
        #print(f'orig logits after softmax: {orig_logits}')

        if not self.config["only_pos_tok"]:
            logits = logits[:, :, 0]  # (bs, 6)

        #print(f'logits after select: {logits.shape}')

        # compute logits and loss
        loss_dict = dict()
        loss = None

        if labels is not None:
            # scaling for labels to be between 0 to 1
            if self.config["loss_fn"] == 'bce':
                labels = (labels - self.lb)/(self.ub-self.lb)

            if self.config["loss_fn"] == 'mcrmse':
                loss = mcrmse_loss_fn(logits, labels)
            else:
                loss = self.loss_fn(logits, labels)

            loss_dict["loss"] = loss

            # target-wise loss computations
            with torch.no_grad():
                for idx, target_name in enumerate(self.target_names):
                    if self.config["loss_fn"] == 'mcrmse':
                        loss_dict[target_name] = mcrmse_loss_fn(
                            logits[:, idx].reshape(-1, 1),
                            labels[:, idx].reshape(-1, 1)
                        )
                    else:
                        loss_dict[target_name] = self.loss_fn(
                            logits[:, idx].reshape(-1, 1),
                            labels[:, idx].reshape(-1, 1)
                        )
        return logits, loss, loss_dict
