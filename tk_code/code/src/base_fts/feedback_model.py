import pdb
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import LayerNorm
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Attention, StableDropout)
from ..utils import mcrmse_loss_fn, RMSELoss, weighted_loss
from collections import OrderedDict

from loguru import logger
#-------- Focal Loss --------------------------------------------------------#


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class PROMPTEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        super(PROMPTEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.parameter.Parameter(self.initialize_embedding(wte,
                                                                                  n_tokens,
                                                                                  random_range,
                                                                                  initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(wte.weight.size(1), n_tokens).uniform_(-random_range, random_range)

    def forward(self, tokens):
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)

class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
        
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


def reinit_deberta(base_model, num_reinit_layers):
    config = base_model.config

    for layer in base_model.encoder.layer[-num_reinit_layers:]:
        for module in layer.modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)


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

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class AttentionBlock(nn.Module):
    def __init__(self, in_features, middle_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.middle_features = middle_features
        self.out_features = out_features
        self.W = nn.Linear(in_features, middle_features)
        self.V = nn.Linear(middle_features, out_features)

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector


class FeedbackModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=True):
        super().__init__()
        self.cfg = cfg
        if cfg["loss_fn"] == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=cfg['l1_loss_beta'])
        elif cfg["loss_fn"] == "rmse":
            self.loss_fn = RMSELoss()
        elif cfg["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss()


        self.aux_loss_fn = nn.BCELoss() #RMSELoss() #nn.MSELoss()

        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg['base_model_path'], output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.base_model = AutoModel.from_pretrained(cfg["base_model_path"], config=self.config)
        else:
            self.base_model = AutoModel(self.config)

        if self.cfg["gradient_checkpointing"]:
            self.base_model.gradient_checkpointing_enable()

        if self.cfg['prompt_tuning']:
            prompt_emb = PROMPTEmbedding(self.base_model.get_input_embeddings(),
                                             n_tokens=20,
                                             initialize_from_vocab=True)
            self.base_model.set_input_embeddings(prompt_emb)


        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size

        if self.cfg['lstm']:
            self.lstm_layer = nn.LSTM(
                input_size=feature_size,
                hidden_size=feature_size//2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        # Embedding types CLS/Mean pool etc
        if self.cfg['embedding'] == 'pool':
            self.pool = MeanPooling()
        elif self.cfg['embedding'] == 'attn':
            self.attn_head = AttentionHead(feature_size, feature_size)
        elif self.cfg['embedding'] == 'mean':
            feature_size = feature_size * 4
        elif self.cfg['embedding'] == 'last-mean-max':
            feature_size = feature_size * 2

        if self.cfg['additional_features']:
            feature_size += self.cfg['additional_features']

        if self.cfg['dropout']:
            self.dropout = nn.Dropout(self.cfg['dropout'])

        # Heads
        if self.cfg['head'] == 'norm':
            self.head = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Linear(feature_size, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, 6)
            )
        elif self.cfg['head'] == 'ln':
            self.head = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Linear(feature_size, 6)
            )
        elif self.cfg['head'] == 'multihead':
            self.head = nn.ModuleList(
                [
                    FeedbackHead(feature_size, 1)
                    for i in range(self.cfg['num_labels'])
                ]
            )
        else:
            self.head = nn.Linear(feature_size, 6)

        if self.cfg['num_layers_reinit'] > 0:
            if 'deb' in self.cfg['base_model_path']:
                reinit_deberta(self.base_model, self.cfg["num_layers_reinit"])

        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                input_ids,
                attention_mask,
                features,
                labels=None,
                **kwargs):

        outputs = self.base_model(input_ids, attention_mask=attention_mask, return_dict=True)
        #logger.debug(f'keys: {outputs.keys()}')
        last_hidden_states = outputs["last_hidden_state"]

        if self.cfg["embedding"] == 'cls':
            last_hidden_states = last_hidden_states[:, 0, :]

        feature = last_hidden_states

        if self.cfg["lstm"]:
            self.lstm_layer.flatten_parameters()
            feature = self.lstm_layer(feature)[0]

        # Mean pool vs CLS
        if self.cfg["embedding"] == 'attn':
            feature = self.attn_head(feature)
        elif self.cfg["embedding"] == 'mean':
            hs = outputs['hidden_states']
            seq_output = torch.cat([hs[-1], hs[-2], hs[-3], hs[-4]], dim=-1)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(seq_output.size()).float()
            sum_embeddings = torch.sum(seq_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            feature = sum_embeddings / sum_mask
        elif self.cfg["embedding"] == 'last-mean':
            feature = torch.mean(feature, axis=1)
        elif self.cfg["embedding"] == 'last-mean-max':
            output = last_hidden_states
            average_pool = torch.mean(output, 1)
            max_pool, _ = torch.max(output, 1)
            feature = torch.cat((average_pool, max_pool), 1)
        elif self.cfg["embedding"] == 'pool':
            feature = self.pool(feature, attention_mask)

        #logger.debug(feature.shape)

        # Add text features to Transformer output before regression head
        if self.cfg['additional_features']:
            feature = torch.cat([feature, features], -1)

        if self.cfg['dropout']:
            feature = self.dropout(feature)

        if self.cfg['head'] == 'multihead':
            logits = [head(feature) for head in self.head]
            output = torch.cat(logits, dim=-1)
        else:
            output = self.head(feature)

        #logger.debug(f'Output shape: {output.shape} labels shape: {labels.shape}')
        #logger.debug(f'Output: {output} labels: {labels}')
        if self.cfg['loss_fn'] == "mcrmse":
            loss = mcrmse_loss_fn(output, labels)
        elif self.cfg["loss_fn"] == "weighted":
            loss = weighted_loss(output, labels)
        else:
            loss = self.loss_fn(output, labels)

        if self.cfg['aux_loss']:
            #aux_loss = self.aux_loss_fn(output, labels)
            #logger.debug(f'outputs: {outputs}')
            #logger.debug(f'labels: {labels}')
            aux_loss = mcrmse_loss_fn(output, labels)
            loss = loss + aux_loss

        return output, loss, None


class FeedbackModelMPL(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=True):
        super().__init__()
        self.cfg = cfg
        if cfg["loss_fn"] == "smooth_l1":
            self.loss_fn = nn.SmoothL1Loss(reduction='mean', beta=cfg['l1_loss_beta'])
        elif cfg["loss_fn"] == "rmse":
            self.loss_fn = RMSELoss()
        elif cfg["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss()

        self.aux_loss_fn = nn.BCELoss() #RMSELoss() #nn.MSELoss()

        if config_path is None:
            self.config = AutoConfig.from_pretrained(cfg['base_model_path'], output_hidden_states=True)
            self.config.hidden_dropout = 0.
            self.config.hidden_dropout_prob = 0.
            self.config.attention_dropout = 0.
            self.config.attention_probs_dropout_prob = 0.
        else:
            self.config = torch.load(config_path)
        if pretrained:
            self.base_model = AutoModel.from_pretrained(cfg["base_model_path"], config=self.config)
        else:
            self.base_model = AutoModel(self.config)

        if self.cfg["gradient_checkpointing"]:
            self.base_model.gradient_checkpointing_enable()

        if self.cfg['prompt_tuning']:
            prompt_emb = PROMPTEmbedding(self.base_model.get_input_embeddings(),
                                             n_tokens=20,
                                             initialize_from_vocab=True)
            self.base_model.set_input_embeddings(prompt_emb)


        hidden_size = self.base_model.config.hidden_size
        feature_size = hidden_size

        if self.cfg['lstm']:
            self.lstm_layer = nn.LSTM(
                input_size=feature_size,
                hidden_size=feature_size//2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        # Embedding types CLS/Mean pool etc
        if self.cfg['embedding'] == 'pool':
            self.pool = MeanPooling()
        elif self.cfg['embedding'] == 'attn':
            self.attn_head = AttentionHead(feature_size, feature_size)
        elif self.cfg['embedding'] == 'mean':
            feature_size = feature_size * 4
        elif self.cfg['embedding'] == 'last-mean-max':
            feature_size = feature_size * 2

        if self.cfg['additional_features']:
            feature_size += self.cfg['additional_features']

        # Heads
        if self.cfg['head'] == 'norm':
            self.head = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Linear(feature_size, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, 6)
            )
        elif self.cfg['head'] == 'ln':
            self.head = nn.Sequential(
                nn.LayerNorm(feature_size),
                nn.Linear(feature_size, 6)
            )
        else:
            self.head = nn.Linear(feature_size, 6)

        self._init_weights(self.head)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self,
                input_ids,
                attention_mask,
                features,
                labels=None,
                **kwargs):

        outputs = self.base_model(input_ids, attention_mask=attention_mask, return_dict=True)
        last_hidden_states = outputs["last_hidden_state"]

        if self.cfg["embedding"] == 'cls':
            last_hidden_states = last_hidden_states[:, 0, :]

        if self.cfg["lstm"]:
            self.lstm_layer.flatten_parameters()
            feature = self.lstm_layer(last_hidden_states)[0]
        else:
            feature = last_hidden_states

        # Mean pool vs CLS
        if self.cfg["embedding"] == 'attn':
            feature = self.attn_head(feature)
        elif self.cfg["embedding"] == 'mean':
            hs = outputs['hidden_states']
            seq_output = torch.cat([hs[-1], hs[-2], hs[-3], hs[-4]], dim=-1)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(seq_output.size()).float()
            sum_embeddings = torch.sum(seq_output * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            feature = sum_embeddings / sum_mask
        elif self.cfg["embedding"] == 'last-mean':
            feature = torch.mean(feature, axis=1)
        elif self.cfg["embedding"] == 'last-mean-max':
            output = last_hidden_states
            average_pool = torch.mean(output, 1)
            max_pool, _ = torch.max(output, 1)
            feature = torch.cat((average_pool, max_pool), 1)
        elif self.cfg["embedding"] == 'pool':
            feature = self.pool(feature, attention_mask)

        #logger.debug(feature.shape)

        # Add text features to Transformer output before regression head
        if self.cfg['additional_features']:
            feature = torch.cat([feature, features], -1)

        output = self.head(feature)


        return output

    def compute_loss(self, logits, labels, **kwargs):
        if self.cfg["loss_fn"] == "bce":
            labels = (labels - 1.0)/4.0
        loss = self.loss_fn(logits, labels)
        return loss


class FeedbackHead(nn.Module):
    """
    classification head with
        - multi-head attention mechanism
        - weighted average of top transformer layers
    """

    def __init__(self, hidden_size, num_targets):
        super(FeedbackHead, self).__init__()
        self.num_targets = num_targets
        #self.classifier = nn.Linear(hidden_size, num_targets)

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, feature):
        # weighted average of layers
        # pdb.set_trace()
        # compute logits
        logits = self.classifier(feature).reshape(-1, self.num_targets)

        return logits

