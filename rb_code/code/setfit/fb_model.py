from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import AutoConfig, AutoModel, BertConfig
from transformers.models.bert.modeling_bert import BertAttention

#-------- Loss function --------------------------------------------------#


class SiameseDistanceMetric(Enum):
    """
    The metric for the contrastive loss
    """
    def EUCLIDEAN(x, y): return F.pairwise_distance(x, y, p=2)
    def MANHATTAN(x, y): return F.pairwise_distance(x, y, p=1)
    def COSINE_DISTANCE(x, y): return 1-F.cosine_similarity(x, y)


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss. Expects as input two texts and a label of either 0 or 1. If the label == 1, then the distance between the
    two embeddings is reduced. If the label == 0, then the distance between the embeddings is increased.
    Further information: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    :param model: SentenceTransformer model
    :param distance_metric: Function that returns a distance between two embeddings. 
    The class SiameseDistanceMetric contains pre-defined matrices that can be used
    :param margin: Negative samples (label == 0) should have a distance of at least the margin value.
    :param size_average: Average by the size of the mini-batch.
    Example::
        from sentence_transformers import SentenceTransformer, LoggingHandler, losses, InputExample
        from torch.utils.data import DataLoader
        model = SentenceTransformer('all-MiniLM-L6-v2')
        train_examples = [
            InputExample(texts=['This is a positive pair', 'Where the distance will be minimized'], label=1),
            InputExample(texts=['This is a negative pair', 'Their distance will be increased'], label=0)]
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=2)
        train_loss = losses.ContrastiveLoss(model=model)
        model.fit([(train_dataloader, train_loss)], show_progress_bar=True)
    """

    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin=0.5, size_average=True):
        super(ContrastiveLoss, self).__init__()
        self.distance_metric = distance_metric
        self.margin = margin
        self.size_average = size_average

    def get_config_dict(self):
        distance_metric_name = self.distance_metric.__name__
        for name, value in vars(SiameseDistanceMetric).items():
            if value == self.distance_metric:
                distance_metric_name = "SiameseDistanceMetric.{}".format(name)
                break

        return {'distance_metric': distance_metric_name, 'margin': self.margin, 'size_average': self.size_average}

    def forward(self, rep_anchor, rep_other, labels):
        # reps = [self.model(sentence_feature)['sentence_embedding'] for sentence_feature in sentence_features]
        # assert len(reps) == 2
        # rep_anchor, rep_other = reps
        distances = self.distance_metric(rep_anchor, rep_other)
        losses = 0.5 * (labels.float() * distances.pow(2) + (1 - labels).float() * F.relu(self.margin - distances).pow(2))
        return losses.mean() if self.size_average else losses.sum()


class CosineSimilarityLoss(nn.Module):
    """
    adapted from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/CosineSimilarityLoss.py
    CosineSimilarityLoss expects, sentence embeddings from two texts and a float labels.
    By default, it minimizes the following loss: ||input_label - cos_score_transformation(cosine_sim(u,v))||_2.

    :param loss_fct: Which pytorch loss function should be used to compare the cosine_similarity(u,v) with the input_label? 
    By default, MSE:  ||input_label - cosine_sim(u,v)||_2
    :param cos_score_transformation: The cos_score_transformation function is applied on top of cosine_similarity. 
    By default, the identify function is used (i.e. no change).
    """

    def __init__(self, loss_fct=nn.MSELoss(), cos_score_transformation=nn.Identity()):
        super(CosineSimilarityLoss, self).__init__()
        self.loss_fct = loss_fct
        self.cos_score_transformation = cos_score_transformation

    def forward(self, feat_1, feat_2, labels):
        output = self.cos_score_transformation(torch.cosine_similarity(feat_1, feat_2))
        return self.loss_fct(output, labels.view(-1))

#-------- Embedding Extractor --------------------------------------------#


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
    feature extraction head with 
        - multi-head attention mechanism + lstm
        - weighted average of top transformer layers
    """

    def __init__(self, hidden_size, num_layers_in_head=12, num_targets=6):
        super(FeatureExtractor, self).__init__()

        self.num_layers_in_head = num_layers_in_head
        self.num_targets = num_targets

        # learnable params weighted average of layers
        init_amax = 5
        weight_data = torch.linspace(-init_amax, init_amax, num_layers_in_head)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)

        # Attention heads
        attention_config = BertConfig()
        attention_config.update(
            {
                "num_attention_heads": 4,
                "hidden_size": hidden_size,
                "attention_probs_dropout_prob": 0.0,
                "hidden_dropout_prob": 0.0,
                "is_decoder": False,
            }
        )
        self.attention = BertAttention(attention_config, position_embedding_type="absolute")
        self.pool = MeanPooling()

        self.classifier = nn.Linear(hidden_size, num_targets)

    def forward(self, backbone_outputs, attention_mask):
        # weighted average of layers
        # pdb.set_trace()

        x = torch.stack(backbone_outputs.hidden_states[-self.num_layers_in_head:])
        w = F.softmax(self.weights, dim=0)
        encoder_layer = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)

        # self.lstm_layer.flatten_parameters()
        # encoder_layer = self.lstm_layer(encoder_layer)[0]

        # apply attention
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]

        # pdb.set_trace()
        context_vector = self.pool(encoder_layer, attention_mask)  # mean pooling -> (bs, h)

        return context_vector
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


#-------- Model ----------------------------------------------------------------------#
class FeedbackSentenceTransformer(nn.Module):
    """
    The feedback-ells sentence transformer model with separate task specific feature extractor
    """

    def __init__(self, config):
        print("initializing the feedback model...")

        super(FeedbackSentenceTransformer, self).__init__()
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

        #----------------------------- Head --------------------------------------------#
        hidden_size = self.backbone.config.hidden_size
        feature_size = hidden_size

        self.feature_extractors = nn.ModuleList(
            [
                FeatureExtractor(
                    hidden_size=feature_size,
                    num_layers_in_head=self.config["num_layers_in_head"],
                    num_targets=1)
                for i in range(self.num_targets)
            ]
        )

        #----------------------------- Loss --------------------------------------------#
        if config["label_mode"] == "cos_sim":
            print(f">>> Model training will use CosineSimilarityLoss")
            self.loss_fn = CosineSimilarityLoss()
        elif config["label_mode"] == "contrastive":
            print(f">>> Model training will use ContrastiveLoss")
            self.loss_fn = ContrastiveLoss()
        else:
            raise NotImplementedError

    def encode(
        self,
        input_ids,
        attention_mask,
    ):
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        features = [extractor(outputs, attention_mask) for extractor in self.feature_extractors]
        embeddings = torch.stack(features, dim=1)  # (batch, 6, hidden_size)
        return embeddings

    def forward(
        self,
        input_ids_sent_1,
        attention_mask_sent_1,
        input_ids_sent_2,
        attention_mask_sent_2,
        labels=None,
        **kwargs
    ):
        outputs_sent_1 = self.backbone(
            input_ids_sent_1,
            attention_mask=attention_mask_sent_1,
            output_hidden_states=True,
        )

        outputs_sent_2 = self.backbone(
            input_ids_sent_2,
            attention_mask=attention_mask_sent_2,
            output_hidden_states=True,
        )

        features_sent_1 = [extractor(outputs_sent_1, attention_mask_sent_1) for extractor in self.feature_extractors]
        embeddings_sent_1 = torch.stack(features_sent_1, dim=1)  # (batch, 6, hidden_size)
        features_sent_2 = [extractor(outputs_sent_2, attention_mask_sent_2) for extractor in self.feature_extractors]
        embeddings_sent_2 = torch.stack(features_sent_2, dim=1)  # (batch, 6, hidden_size)

        # compute logits and loss
        loss_dict = dict()
        embeddings = dict()
        loss = None

        embeddings["sent_1"] = embeddings_sent_1
        embeddings["sent_2"] = embeddings_sent_2

        if labels is not None:
            for idx, target_name in enumerate(self.target_names):
                if self.config["label_mode"] == "contrastive":
                    labels = labels.long()  # cast to integer

                target_loss = self.loss_fn(
                    features_sent_1[idx],
                    features_sent_2[idx],
                    labels[:, idx]
                )
                if idx == 0:
                    loss = target_loss
                else:
                    loss += target_loss

                loss_dict[target_name] = torch.clone(target_loss.detach())

        return embeddings, loss, loss_dict

#-------- Feedback SetFit ----------------------------------------------------------------------#


class FeedbackModelSetFit(nn.Module):
    """
    The feedback-ells model with separate task specific heads
    """

    def __init__(self, config):
        print("initializing the setfit feedback model...")

        super(FeedbackModelSetFit, self).__init__()
        assert type(config) == dict

        #----------------------------- Sentence Transformer ------------------------------------#
        self.sentence_transformer = FeedbackSentenceTransformer(config)

        # print(config)
        config = config["model"]
        self.config = config

        # enable gradient checkpointing
        self.sentence_transformer.backbone.gradient_checkpointing_enable()

        # print("loading sentence transformer from previous checkpoint...")
        # checkpoint = config["model"]["ckpt_path"]
        # ckpt = torch.load(checkpoint)
        # self.sentence_transformer.load_state_dict(ckpt['state_dict'])
        # del ckpt
        # gc.collect()

        hidden_size = self.sentence_transformer.backbone.config.hidden_size
        #----------------------------- Fusion --------------------------------------------------#
        # Attention heads
        if config["use_fusion"]:
            attention_config = BertConfig()
            attention_config.update(
                {
                    "num_attention_heads": 4,
                    "hidden_size": hidden_size,
                    "attention_probs_dropout_prob": 0.0,
                    "hidden_dropout_prob": 0.0,
                    "is_decoder": False,
                }
            )
            self.fusion = BertAttention(attention_config, position_embedding_type=None)
        # self.pool = MeanPooling()

        #----------------------------- Classifier  ---------------------------------------------#
        self.lb = 1.0  # lowest score
        self.ub = 5.0  # highest score

        self.target_names = config["target_names"]
        self.num_targets = len(self.target_names)

        if config["flatten_embeddings"]:
            self.classifier = nn.Linear(hidden_size*self.num_targets, self.num_targets)
        else:
            self.classifiers = nn.ModuleList([nn.Linear(hidden_size, 1) for i in range(self.num_targets)])

        #----------------------------- Loss ---------------------------------------------------#
        print(f">>> Model training will use <{config['loss_fn']}> loss function")
        if config["loss_fn"] == "mse":
            self.loss_fn = nn.MSELoss(reduction='mean')
        elif config["loss_fn"] == "bce":
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')
        else:
            raise NotImplementedError

    def freeze(self, n_layers, freeze_all=False):
        # freeze embeddings & layers
        if freeze_all:
            print("freezing entire backbone...")
            self.sentence_transformer.backbone.embeddings.requires_grad_(False)
            self.sentence_transformer.backbone.encoder.layer[:].requires_grad_(False)
            self.sentence_transformer.feature_extractors[:].requires_grad_(False)
        else:
            print(f"setting requires grad to false for first {n_layers} layers")
            self.sentence_transformer.backbone.embeddings.requires_grad_(False)
            self.sentence_transformer.backbone.encoder.layer[:n_layers].requires_grad_(False)

    def unfreeze(self, n_layers, unfreeze_all=False):
        # unfreeze embeddings & layers
        if unfreeze_all:
            print("unfreezing entire backbone...")
            self.sentence_transformer.feature_extractors[:].requires_grad_(True)
            self.sentence_transformer.backbone.encoder.layer[:].requires_grad_(True)
            self.sentence_transformer.backbone.embeddings.requires_grad_(True)

        else:
            print(f"setting requires grad to true for last {n_layers} layers")
            self.sentence_transformer.feature_extractors[:].requires_grad_(True)
            self.sentence_transformer.backbone.encoder.layer[-n_layers:].requires_grad_(True)

    def forward(
        self,
        input_ids,
        attention_mask,
        labels=None,
        **kwargs
    ):
        # extract embeddings  -> (bs, num_targets, hidden_size)
        bs = input_ids.size(0)
        embeddings = self.sentence_transformer.encode(input_ids, attention_mask)  # (bs, 6, hidden_size)

        if self.config["use_fusion"]:
            # extended_attention_mask = attention_mask[:, None, None, :]
            # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
            embeddings = self.fusion(embeddings)[0]  # , extended_attention_mask)[0]

        if self.config["flatten_embeddings"]:
            embeddings = embeddings.reshape(bs, -1)  # (bs, hidden_size*6)
            logits = self.classifier(embeddings)
        else:
            logits = [classifier(embeddings[:, idx]) for idx, classifier in enumerate(self.classifiers)]
            logits = torch.cat(logits, dim=-1)  # (bs, 6)

        # logits = self.classifier(embeddings)  # (bs, hidden_size, num_targets)

        # compute logits and loss
        loss_dict = dict()
        loss = None

        # if labels is not None:
        #     if self.config["loss_fn"] == "bce":
        #         # scaling for labels to be between 0 to 1
        #         labels = (labels - self.lb)/(self.ub-self.lb)

        #     loss = self.loss_fn(logits, labels)
        #     loss_dict["loss"] = loss

        #     # target-wise loss computations
        #     with torch.no_grad():
        #         for idx, target_name in enumerate(self.target_names):
        #             loss_dict[target_name] = self.loss_fn(
        #                 logits[:, idx].reshape(-1, 1),
        #                 labels[:, idx].reshape(-1, 1)
        #             )

        # weighted loss
        loss_weights = [
            1.05,  # cohesion
            1.00,  # syntax
            0.95,  # vocabulary
            0.95,  # phraseology
            1.00,  # grammar
            0.95,  # conventions
        ]

        if labels is not None:
            for idx, target_name in enumerate(self.target_names):
                if self.config["loss_fn"] == "bce":
                    # scaling for labels to be between 0 to 1
                    labels = (labels - self.lb)/(self.ub-self.lb)

                target_loss = self.loss_fn(
                    logits[:, idx].reshape(-1, 1),
                    labels[:, idx].reshape(-1, 1),
                )

                if idx == 0:
                    loss = loss_weights[idx]*target_loss/self.num_targets
                else:
                    loss += loss_weights[idx]*target_loss/self.num_targets

                loss_dict[target_name] = torch.clone(target_loss.detach())

        return logits, loss, loss_dict
