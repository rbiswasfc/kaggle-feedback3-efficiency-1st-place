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


def get_aux_loss(features, labels, temp=0.05):
    feat_1 = features.unsqueeze(1)
    feat_2 = features.unsqueeze(0)
    logits = F.cosine_similarity(feat_1, feat_2, dim=-1) / temp

    labels1 = labels.unsqueeze(1)
    labels2 = labels.unsqueeze(0)

    # sim_scores = 1. - -2.0*torch.abs(labels1.float()-labels2.float())/8.0
    # loss = nn.MSELoss()(logits.view(-1), sim_scores.view(-1))

    matches = (labels1 == labels2).byte()
    loss = nn.BCEWithLogitsLoss()(logits.view(-1), matches.view(-1).float())
    return loss
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Feature Extractor
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()

    def forward(self, last_hidden_state, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings


class FeatureExtractor(nn.Module):
    """
    extract features from backbone outputs 
        - multi-head attention mechanism
        - weighted average of top transformer layers
    """

    def __init__(self, config):
        super(FeatureExtractor, self).__init__()

        self.config = config
        self.num_layers = config["num_layers"]
        self.num_features = config["hidden_size"]

        #------------ weighted-average ---------------------------------------------------#
        init_amax = 5
        weight_data = torch.linspace(-init_amax, init_amax, self.num_layers)
        # weight_data = torch.tensor([1] * self.num_layers, dtype=torch.float)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)

        #------------ multi-head attention -----------------------------------------------#
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

        #------------ mean-pooling ------------------------------------------------------#
        self.pool = MeanPooling()

        #------------ Layer Normalization ------------------------------------------------#
        self.layer_norm = nn.LayerNorm(self.num_features, 1e-7)

    def forward(self, backbone_outputs, attention_mask):

        #------------ Output Transformation ----------------------------------------------#
        x = torch.stack(backbone_outputs.hidden_states[-self.num_layers:])
        w = F.softmax(self.weights, dim=0)
        encoder_layer = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)

        #------------ Multi-head attention  ----------------------------------------------#
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_layer = self.mha_layer_norm(encoder_layer)
        encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]

        #------------ mean-pooling  ------------------------------------------------------#
        features = self.pool(encoder_layer, attention_mask)  # mean pooling

        #------------ layer-normalization  -----------------------------------------------#
        features = self.layer_norm(features)  # (bs, num_features)

        return features

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Model
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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


class FeedbackModel(nn.Module):
    """
    The feedback-ells model with separate task specific heads
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
                "use_cache": False,
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
            print(f"setting requires grad to false for first {config['n_freeze']} layers")
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.layer[:config["n_freeze"]].requires_grad_(False)

        #----------------------------- Feature Extractor ---------------------------------#
        hidden_size = num_features = self.backbone.config.hidden_size
        config["feature_extractor"]["hidden_size"] = hidden_size

        self.feature_extractors = nn.ModuleList(
            [
                FeatureExtractor(config["feature_extractor"]) for i in range(self.num_targets)
            ]
        )

        #----------------------------- Classifiers -------------------------------------#
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(num_features, 1) for i in range(self.num_targets)
            ]
        )

        #----------------------------- Loss --------------------------------------------#
        print(f">>> Model training will use <<{config['loss_fn']}>> loss function")
        if config["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction='none')
        elif config["loss_fn"] == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        elif config["loss_fn"] == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=1.0)
        else:
            raise NotImplementedError

    def encode(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        features = [extractor(outputs, attention_mask) for extractor in self.feature_extractors]
        embeddings = torch.stack(features, dim=1)  # (batch, num_targets, num_features)

        return embeddings

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
        aux_labels=None,
        **kwargs
    ):

        # features
        features = self.encode(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # (bs, num_targets, num_features)

        # logits
        logits = [classifier(features[:, idx]) for idx, classifier in enumerate(self.classifiers)]
        logits = torch.cat(logits, dim=-1)  # (bs, num_features)

        # loss
        loss_dict = dict()
        loss = None

        if labels is not None:
            if self.config["loss_fn"] == "bce":
                # scaling for labels to be between 0 to 1
                labels = (labels - self.lb)/(self.ub-self.lb)

            loss = self.loss_fn(logits, labels)

            # contrastive loss
            if self.config["use_contrastive_loss"]:
                contrastive_losses = [get_aux_loss(features[:, idx], aux_labels[:, idx]) for idx in range(self.num_targets)]
                contrastive_loss = torch.mean(torch.tensor(contrastive_losses))
                loss_dict["contrastive_loss"] = contrastive_loss
                contrastive_weight = 0.15
                loss = loss + contrastive_weight*contrastive_loss

            if self.config["use_weighted_loss"]:
                max_l, _ = torch.max(labels, dim=-1)
                # print(max_l)
                # print(max_l.shape)
                min_l, _ = torch.min(labels, dim=-1)
                range_l = max_l - min_l
                weights = 1.0 + range_l
                # print(weights)
                # print(weights.shape)
                # print(loss.shape)

                loss = (weights.unsqueeze(-1)*loss).mean()
            else:
                loss = loss.mean()

            loss_dict["loss"] = loss

            # target-wise loss computations
            with torch.no_grad():
                for idx, target_name in enumerate(self.target_names):
                    loss_dict[target_name] = self.loss_fn(
                        logits[:, idx].reshape(-1, 1),
                        labels[:, idx].reshape(-1, 1)
                    ).mean()

        return logits, loss, loss_dict

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Meta-PL
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class FeedbackModelMPL(nn.Module):
    """
    The feedback-ells model with separate task specific heads
    """

    def __init__(self, config):
        print("initializing the feedback model...")

        super(FeedbackModelMPL, self).__init__()
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
                "use_cache": False,
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
            print(f"setting requires grad to false for first {config['n_freeze']} layers")
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.layer[:config["n_freeze"]].requires_grad_(False)

        #----------------------------- Feature Extractor ---------------------------------#
        hidden_size = num_features = self.backbone.config.hidden_size
        config["feature_extractor"]["hidden_size"] = hidden_size

        self.feature_extractors = nn.ModuleList(
            [
                FeatureExtractor(config["feature_extractor"]) for i in range(self.num_targets)
            ]
        )

        #----------------------------- Classifiers -------------------------------------#
        self.classifiers = nn.ModuleList(
            [
                nn.Linear(num_features, 1) for i in range(self.num_targets)
            ]
        )

        #----------------------------- Loss --------------------------------------------#
        print(f">>> Model training will use <<{config['loss_fn']}>> loss function")
        if config["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError

    def encode(self, input_ids, attention_mask, token_type_ids):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )
        features = [extractor(outputs, attention_mask) for extractor in self.feature_extractors]
        embeddings = torch.stack(features, dim=1)  # (batch, num_targets, num_features)

        return embeddings

    def get_logits(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        **kwargs,
    ):

        # features
        features = self.encode(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )  # (bs, num_targets, num_features)

        # logits
        logits = [classifier(features[:, idx]) for idx, classifier in enumerate(self.classifiers)]
        logits = torch.cat(logits, dim=-1)  # (bs, num_features)

        return logits

    def compute_loss(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        return loss

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# AWP
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


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
