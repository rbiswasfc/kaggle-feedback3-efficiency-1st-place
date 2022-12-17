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

#------------------------- Loss -------------------------------------------------------#


def get_ranking_loss(logits, labels, margin=0.0):
    labels1 = labels.unsqueeze(1)
    labels2 = labels.unsqueeze(0)

    logits1 = logits.unsqueeze(1)
    logits2 = logits.unsqueeze(0)

    y_ij = labels1 - labels2
    r_ij = (logits1 >= logits2).float() - (logits1 < logits2).float()

    residual = torch.clamp(-r_ij*y_ij + margin, min=0.0)
    loss = residual.mean()
    return loss


def get_sim_loss(logits, labels):
    similarity = F.cosine_similarity(labels, logits, dim=-1)
    loss = 1.0 - similarity
    return loss


def get_bt_loss(logits, labels):
    bs = len(labels)
    num_comparisons = bs * (bs - 1) / 2

    true_comparison = labels.unsqueeze(1) - labels.unsqueeze(0)
    pred_comparison = logits.unsqueeze(1) - logits.unsqueeze(0)
    loss = torch.log(1 + torch.tril(torch.exp(-true_comparison * pred_comparison))).sum()
    loss = loss/num_comparisons
    return loss


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Poolers
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


class ContextPooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        context_token = hidden_states[:, 0]
        pooled_output = self.dense(context_token)
        pooled_output = F.relu(pooled_output)  # TODO: check which activation to use
        return pooled_output

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Multi-scale Head
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class MultiScaleHead(nn.Module):
    """
    extract features from backbone outputs 
        - multi-head attention mechanism
        - weighted average of top transformer layers
        - multi-scale essay representations
    """

    def __init__(self, config):
        super(MultiScaleHead, self).__init__()

        self.config = config
        self.num_features = config["hidden_size"]
        self.num_layers = config["num_layers"]

        ###################################################################################
        ###### Common-Block ###############################################################
        ###################################################################################

        init_amax = 5
        weight_data = torch.linspace(-init_amax, init_amax, self.num_layers)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)

        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": 4,
                "hidden_size": self.num_features,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "is_decoder": False,
            }
        )
        self.mha_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.attention = BertAttention(attention_config, position_embedding_type="absolute")

        ###################################################################################
        ###### Paragraph Scale ############################################################
        ###################################################################################

        paragraph_attention_config = BertConfig()
        paragraph_attention_config.update(
            {
                "num_attention_heads": 4,  # 4,
                "hidden_size": self.num_features,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "is_decoder": False,
            }
        )
        self.paragraph_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.paragraph_attention = BertAttention(paragraph_attention_config, position_embedding_type="absolute")

        self.paragraph_lstm_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.paragraph_lstm_layer = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.num_features//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        ###################################################################################
        ###### Sentence Scale #############################################################
        ###################################################################################

        sentence_attention_config = BertConfig()
        sentence_attention_config.update(
            {
                "num_attention_heads": 4,  # 4,
                "hidden_size": self.num_features,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "is_decoder": False,
            }
        )
        self.sentence_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.sentence_attention = BertAttention(sentence_attention_config, position_embedding_type="absolute")

        self.sentence_lstm_layer_norm = nn.LayerNorm(self.num_features, 1e-7)
        self.sentence_lstm_layer = nn.LSTM(
            input_size=self.num_features,
            hidden_size=self.num_features//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        ###################################################################################
        ###### Pooling ####################################################################
        ###################################################################################
        self.pool = MeanPooling()
        self.context_pool = ContextPooler(self.num_features)

        ###################################################################################
        ###### Classifiers ################################################################
        ###################################################################################
        self.context_pool_classifier = nn.Linear(self.num_features, 1)
        self.mean_pool_classifier = nn.Linear(self.num_features, 1)
        self.paragraph_pool_classifier = nn.Linear(self.num_features, 1)
        self.sentence_pool_classifier = nn.Linear(self.num_features, 1)

    def forward(
        self,
        backbone_outputs,
        attention_mask,
        paragraph_head_idxs,
        paragraph_tail_idxs,
        paragraph_attention_mask,
        sentence_head_idxs,
        sentence_tail_idxs,
        sentence_attention_mask,
    ):

        ###################################################################################
        ###### Scorer 1: Context Pool #####################################################
        ###################################################################################
        context_features = backbone_outputs[0]
        context_features = self.context_pool(context_features)
        context_logits = self.context_pool_classifier(context_features)

        ###################################################################################
        ###### Scorer 2: Mean Pool ########################################################
        ###################################################################################

        x = torch.stack(backbone_outputs.hidden_states[-self.num_layers:])
        w = F.softmax(self.weights, dim=0)
        encoder_layer = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)

        encoder_layer = self.mha_layer_norm(encoder_layer)
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]

        mean_pool_features = self.pool(encoder_layer, attention_mask)
        mean_pool_logits = self.context_pool_classifier(mean_pool_features)

        ###################################################################################
        ###### Scorer 3: Paragraph Scale ##################################################
        ###################################################################################

        bs = encoder_layer.shape[0]
        paragraph_feature_vector = []

        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(paragraph_head_idxs[i], paragraph_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
            paragraph_feature_vector.append(span_vec_i)

        paragraph_feature_vector = torch.stack(paragraph_feature_vector)  # (bs, num_spans, h)
        paragraph_feature_vector = self.paragraph_layer_norm(paragraph_feature_vector)  # (bs, num_spans, h)

        self.paragraph_lstm_layer.flatten_parameters()
        paragraph_feature_vector = self.paragraph_lstm_layer(paragraph_feature_vector)[0]

        extended_attention_mask = paragraph_attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        paragraph_feature_vector = self.paragraph_attention(paragraph_feature_vector, extended_attention_mask)[0]

        paragraph_feature_vector = self.pool(paragraph_feature_vector, paragraph_attention_mask)
        paragraph_logits = self.paragraph_pool_classifier(paragraph_feature_vector)

        ###################################################################################
        ###### Scorer 4: Sentence Scale ###################################################
        ###################################################################################

        bs = encoder_layer.shape[0]
        sentence_feature_vector = []

        for i in range(bs):
            span_vec_i = []
            for head, tail in zip(sentence_head_idxs[i], sentence_tail_idxs[i]):
                tmp = torch.mean(encoder_layer[i, head+1:tail], dim=0)  # [h]
                span_vec_i.append(tmp)
            span_vec_i = torch.stack(span_vec_i)  # (num_spans, h)
            sentence_feature_vector.append(span_vec_i)

        sentence_feature_vector = torch.stack(sentence_feature_vector)  # (bs, num_spans, h)
        sentence_feature_vector = self.sentence_layer_norm(sentence_feature_vector)  # (bs, num_spans, h)

        self.sentence_lstm_layer.flatten_parameters()
        sentence_feature_vector = self.sentence_lstm_layer(sentence_feature_vector)[0]

        extended_attention_mask = sentence_attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        sentence_feature_vector = self.sentence_attention(sentence_feature_vector, extended_attention_mask)[0]

        sentence_feature_vector = self.pool(sentence_feature_vector, sentence_attention_mask)
        sentence_logits = self.sentence_pool_classifier(sentence_feature_vector)

        ###################################################################################
        ###### Multi-Scale Ensemble #######################################################
        ###################################################################################
        logits = context_logits + mean_pool_logits + paragraph_logits + sentence_logits
        logits = logits.reshape(bs, 1)
        return logits

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
            print(f"setting requires grad to false for first {config['n_freeze']} layers")
            self.backbone.embeddings.requires_grad_(False)
            self.backbone.encoder.layer[:config["n_freeze"]].requires_grad_(False)

        #----------------------------- Heads -------------------------------------------#
        hidden_size = self.backbone.config.hidden_size
        config["head"]["hidden_size"] = hidden_size
        self.heads = nn.ModuleList([MultiScaleHead(config["head"]) for i in range(self.num_targets)])

        #----------------------------- Loss --------------------------------------------#
        print(f">>> Model training will use <<{config['loss_fn']}>> loss function")
        if config["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        else:
            raise NotImplementedError

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        span_head_idxs,
        span_tail_idxs,
        span_attention_mask,
        sentence_head_idxs,
        sentence_tail_idxs,
        sentence_attention_mask,
        labels=None,
        aux_labels=None,
        **kwargs
    ):

        # backbone outputs
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )

        # logits
        logits = [
            head(
                outputs,
                attention_mask,
                span_head_idxs,
                span_tail_idxs,
                span_attention_mask,
                sentence_head_idxs,
                sentence_tail_idxs,
                sentence_attention_mask,
            ) for head in self.heads
        ]
        logits = torch.cat(logits, dim=-1)  # (bs, num_features)

        # loss
        loss_dict = dict()
        loss = None

        if labels is not None:
            loss = self.loss_fn(logits, labels)
            loss_dict["loss"] = loss

            if self.config.get("use_sim_loss", False):
                sim_losses = [get_sim_loss(logits[:, idx].reshape(-1), labels[:, idx].reshape(-1))
                              for idx in range(self.num_targets)]
                sim_loss = torch.mean(torch.tensor(sim_losses))
                loss_dict["sim_loss"] = sim_loss
                loss += 1.0*sim_loss

            if self.config.get("use_ranking_loss", False):
                ranking_losses = [get_ranking_loss(logits[:, idx].reshape(-1), labels[:, idx].reshape(-1))
                                  for idx in range(self.num_targets)]
                ranking_loss = torch.mean(torch.tensor(ranking_losses))
                loss_dict["ranking_loss"] = ranking_loss
                loss += 1.0*ranking_loss

            if self.config.get("use_bt_loss", False):
                bt_losses = [get_bt_loss(logits[:, idx].reshape(-1), labels[:, idx].reshape(-1))
                                  for idx in range(self.num_targets)]
                bt_loss = torch.mean(torch.tensor(bt_losses))
                loss_dict["bt_loss"] = bt_loss
                loss += 1.0*bt_loss

            # target-wise loss computations
            with torch.no_grad():
                for idx, target_name in enumerate(self.target_names):
                    loss_dict[target_name] = self.loss_fn(
                        logits[:, idx].reshape(-1, 1),
                        labels[:, idx].reshape(-1, 1)
                    )

        return logits, loss, loss_dict


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
