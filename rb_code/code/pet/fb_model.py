import pdb
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention
from transformers.models.deberta.modeling_deberta import DebertaOnlyMLMHead
from transformers.models.deberta_v2.modeling_deberta_v2 import \
    DebertaV2OnlyMLMHead
from transformers.models.electra.modeling_electra import (
    ElectraGeneratorPredictions, ElectraPreTrainedModel)
from transformers.models.roberta.modeling_roberta import RobertaLMHead


class ElectraHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.generator_predictions = ElectraGeneratorPredictions(config)
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, generator_hidden_states):
        prediction_scores = self.generator_predictions(generator_hidden_states)
        prediction_scores = self.generator_lm_head(prediction_scores)
        return prediction_scores

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


class FeedbackModel(nn.Module):
    """
    The feedback-ells model for prompt based fine-tuning
    """

    def __init__(self, config):
        print("initializing the feedback model...")

        super(FeedbackModel, self).__init__()
        config = config["model"]
        self.config = config
        self.target_names = config["target_names"]
        self.num_targets = len(self.target_names)

        self.lb = 1.0  # lowest score
        self.ub = 5.0  # highest score

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

        #------------ weighted-average ---------------------------------------------------#
        if config["use_weighted_average"]:
            init_amax = 5
            weight_data = torch.linspace(-init_amax, init_amax, self.config["num_layers_in_head"])
            weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.weights = nn.Parameter(weight_data, requires_grad=True)

        #------------ multi-head-attention ------------------------------------------------#
        self.num_features = self.backbone.config.hidden_size

        if config["use_multihead_attention"]:
            attention_config = BertConfig()
            attention_config.update(
                {
                    "num_attention_heads": 4,  # 4,
                    "hidden_size": self.num_features,
                    "attention_probs_dropout_prob": 0.0,
                    "hidden_dropout_prob": 0.0,
                    "is_decoder": False,
                }
            )
            self.mha_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
            self.attention = BertAttention(attention_config, position_embedding_type="absolute")

        #----------------------------- Head --------------------------------------------#
        if config["lm_arch"] == "debertaV2":
            print("using DebertaV2 MLM Head")
            self.classifiers = nn.ModuleList([DebertaV2OnlyMLMHead(backbone_config) for _ in range(self.num_targets)])
        elif config["lm_arch"] == "debertaV1":
            print("using DebertaV2 MLM Head")
            self.classifiers = nn.ModuleList([DebertaOnlyMLMHead(backbone_config) for _ in range(self.num_targets)])
        elif config["lm_arch"] == "electra":
            print("using electra MLM head")
            self.classifiers = nn.ModuleList([ElectraHead(backbone_config) for _ in range(self.num_targets)])
        elif config["lm_arch"] == "roberta":
            print("using roberta MLM head")
            self.classifiers = nn.ModuleList([RobertaLMHead(backbone_config) for _ in range(self.num_targets)])
        else:
            raise NotImplementedError

        #----------------------------- Loss --------------------------------------------#
        print(f">>> Model training will use BCEWithLogitsLoss loss function")
        self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        mask_token_idxs=None,
        labels=None,
        pos_tok_ids=None,
        neg_tok_ids=None,
        **kwargs
    ):
        backbone_outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        #------------ Output Transformation ----------------------------------------------#
        if self.config["use_weighted_average"]:
            x = torch.stack(backbone_outputs.hidden_states[-self.config["num_layers_in_head"]:])
            w = F.softmax(self.weights, dim=0)
            sequence_output = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)
        else:
            sequence_output = backbone_outputs[0]

        #------------ multi-head-attention -----------------------------------------------#
        if self.config["use_multihead_attention"]:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            sequence_output = self.mha_layer_norm(sequence_output)
            sequence_output = self.attention(sequence_output, extended_attention_mask)[0]

        mask_token_idxs = mask_token_idxs[0]  # same for all examples in the batch
        sequence_output = sequence_output[:, mask_token_idxs]  # (bs, 6, h)

        # prediction_scores = self.classifier(sequence_output)  # (bs, 6, num_vocab)

        prediction_scores = [classifier(sequence_output[:, target_idx])
                             for target_idx, classifier in enumerate(self.classifiers)]  # [(bs, num_vocab), (bs, num_vocal)...]
        prediction_scores = torch.stack(prediction_scores, dim=1)  # (bs, 6, num_vocab)

        pos_scores = torch.mean(prediction_scores[:, :, pos_tok_ids], dim=-1).unsqueeze(-1)  # (bs, 6, 1)
        neg_scores = torch.mean(prediction_scores[:, :, neg_tok_ids], dim=-1).unsqueeze(-1)  # (bs, 6, 1)

        logits = torch.cat([pos_scores, neg_scores], dim=-1)  # (bs, 6, 2)
        logits = F.softmax(logits, dim=-1)  # (bs, 6, 2)
        logits = logits[:, :, 0]  # (bs, 6)

        # compute logits and loss
        loss_dict = dict()
        loss = None

        if labels is not None:
            # scaling for labels to be between 0 to 1
            labels = (labels - self.lb)/(self.ub-self.lb)
            loss = self.loss_fn(logits, labels)
            loss_dict["loss"] = loss

            # target-wise loss computations
            with torch.no_grad():
                for idx, target_name in enumerate(self.target_names):
                    loss_dict[target_name] = self.loss_fn(
                        logits[:, idx].reshape(-1, 1),
                        labels[:, idx].reshape(-1, 1)
                    )

        return logits, loss, loss_dict


class FeedbackModelAdaPET(nn.Module):
    """
    The feedback-ells model for prompt based fine-tuning
    """

    def __init__(self, config):
        print("initializing the feedback model...")

        super(FeedbackModelAdaPET, self).__init__()
        config = config["model"]
        self.config = config
        self.target_names = config["target_names"]
        self.num_targets = len(self.target_names)

        self.lb = 1.0  # lowest score
        self.ub = 5.0  # highest score

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

        #------------ weighted-average ---------------------------------------------------#
        if config["use_weighted_average"]:
            init_amax = 5
            weight_data = torch.linspace(-init_amax, init_amax, self.config["num_layers_in_head"])
            weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            self.weights = nn.Parameter(weight_data, requires_grad=True)

        #------------ multi-head-attention ------------------------------------------------#
        self.num_features = self.backbone.config.hidden_size

        if config["use_multihead_attention"]:
            attention_config = BertConfig()
            attention_config.update(
                {
                    "num_attention_heads": 4,  # 4,
                    "hidden_size": self.num_features,
                    "attention_probs_dropout_prob": 0.0,
                    "hidden_dropout_prob": 0.0,
                    "is_decoder": False,
                }
            )
            self.mha_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
            self.attention = BertAttention(attention_config, position_embedding_type="absolute")

        #----------------------------- Head --------------------------------------------#
        # if config["use_debv3"]:
        #     print("using DebertaV2 MLM Head")
        #     self.classifiers = nn.ModuleList([DebertaV2OnlyMLMHead(backbone_config) for _ in range(self.num_targets)])
        # else:
        #     print("using electra MLM head")
        #     self.classifiers = nn.ModuleList([ElectraHead(backbone_config) for _ in range(self.num_targets)])

        if config["lm_arch"] == "debertaV2":
            print("using DebertaV2 MLM Head")
            self.classifiers = nn.ModuleList([DebertaV2OnlyMLMHead(backbone_config) for _ in range(self.num_targets)])
        elif config["lm_arch"] == "debertaV1":
            print("using DebertaV2 MLM Head")
            self.classifiers = nn.ModuleList([DebertaOnlyMLMHead(backbone_config) for _ in range(self.num_targets)])
        elif config["lm_arch"] == "electra":
            print("using electra MLM head")
            self.classifiers = nn.ModuleList([ElectraHead(backbone_config) for _ in range(self.num_targets)])
        else:
            raise NotImplementedError

        #----------------------------- Loss --------------------------------------------#
        print(f">>> Model training will use BCEWithLogitsLoss loss function")
        self.loss_fn = nn.BCELoss(reduction='mean')

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        mask_token_idxs=None,
        labels=None,
        pos_tok_ids=None,
        neg_tok_ids=None,
        **kwargs
    ):
        backbone_outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        #------------ Output Transformation ----------------------------------------------#
        if self.config["use_weighted_average"]:
            x = torch.stack(backbone_outputs.hidden_states[-self.config["num_layers_in_head"]:])
            w = F.softmax(self.weights, dim=0)
            sequence_output = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)
        else:
            sequence_output = backbone_outputs[0]

        #------------ multi-head-attention -----------------------------------------------#
        if self.config["use_multihead_attention"]:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            sequence_output = self.mha_layer_norm(sequence_output)
            sequence_output = self.attention(sequence_output, extended_attention_mask)[0]

        mask_token_idxs = mask_token_idxs[0]  # same for all examples in the batch
        sequence_output = sequence_output[:, mask_token_idxs]  # (bs, 6, h)

        # prediction_scores = self.classifier(sequence_output)  # (bs, 6, num_vocab)

        prediction_scores = [classifier(sequence_output[:, target_idx])
                             for target_idx, classifier in enumerate(self.classifiers)]  # [(bs, num_vocab), (bs, num_vocal)...]
        prediction_scores = torch.stack(prediction_scores, dim=1)  # (bs, 6, num_vocab)

        # use softmax
        prediction_scores = F.softmax(prediction_scores, dim=-1)   # (bs, 6, num_vocab)

        logits = torch.sum(prediction_scores[:, :, pos_tok_ids], dim=-1)  # (bs, 6)
        neg_logits = torch.sum(prediction_scores[:, :, neg_tok_ids], dim=-1)  # (bs, 6)

        # logits = torch.cat([pos_scores, neg_scores], dim=-1)  # (bs, 6, 2)
        # logits = F.softmax(logits, dim=-1)  # (bs, 6, 2)
        # logits = logits[:, :, 0]  # (bs, 6)

        # compute logits and loss
        loss_dict = dict()
        loss = None

        if labels is not None:
            # scaling for labels to be between 0 to 1
            labels = (labels - self.lb)/(self.ub-self.lb)
            neg_labels = 1.0 - labels

            pos_loss = self.loss_fn(logits, labels)
            neg_loss = self.loss_fn(neg_logits, neg_labels)
            loss = pos_loss + neg_loss

            loss_dict["loss"] = loss

            # target-wise loss computations
            with torch.no_grad():
                for idx, target_name in enumerate(self.target_names):
                    loss_dict[target_name] = self.loss_fn(
                        logits[:, idx].reshape(-1, 1),
                        labels[:, idx].reshape(-1, 1)
                    )

        return logits, loss, loss_dict
