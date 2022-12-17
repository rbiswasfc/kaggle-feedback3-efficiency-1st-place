"""
@created by: heyao
@created at: 2022-11-10 16:08:52
from: https://www.kaggle.com/code/aerdem4/xgb-lgb-feedback-prize-cv-0-7322/notebook
"""
import numpy as np
from omegaconf import DictConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from transformers import AutoModel, AutoConfig

from feedback_ell.modules.base import BaseLightningModule
from feedback_ell.nn.poolers import CLSPooling, MeanPooling, AttentionPooling, MultiPooling
from feedback_ell.utils import label_columns
from feedback_ell.utils.meter import AverageMeter


class ResidualLSTM(nn.Module):
    def __init__(self, d_model, rnn):
        super(ResidualLSTM, self).__init__()
        self.downsample = nn.Linear(d_model, d_model // 2)
        if rnn == 'GRU':
            self.LSTM = nn.GRU(d_model // 2, d_model // 2, num_layers=2, bidirectional=False, dropout=0.2)
        else:
            self.LSTM = nn.LSTM(d_model // 2, d_model // 2, num_layers=2, bidirectional=False, dropout=0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.norm1 = nn.LayerNorm(d_model // 2)
        self.linear1 = nn.Linear(d_model // 2, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout2 = nn.Dropout(0.2)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        res = x
        x = self.downsample(x)
        x, _ = self.LSTM(x)
        x = self.dropout1(x)
        x = self.norm1(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = self.dropout2(x)
        x = res + x
        return self.norm2(x)


class SlidingWindowTransformerModel(nn.Module):
    def __init__(self, DOWNLOADED_MODEL_PATH, rnn="LSTM", window_size=512, edge_len=64):
        super(SlidingWindowTransformerModel, self).__init__()
        config_model = AutoConfig.from_pretrained(DOWNLOADED_MODEL_PATH + '/config.json')

        self.backbone = AutoModel.from_pretrained(
            DOWNLOADED_MODEL_PATH + '/pytorch_model.bin', config=config_model)

        # self.lstm = ResidualLSTM(config_model.hidden_size, rnn)
        self.lstm = nn.LSTM(config_model.hidden_size, config_model.hidden_size // 2, bidirectional=True, batch_first=False)
        self.window_size = window_size
        self.edge_len = edge_len
        self.inner_len = window_size - edge_len * 2

    def forward(self, input_ids, attention_mask):
        B, L = input_ids.shape

        if L <= self.window_size:
            x = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)[0]
            # pass
        else:
            # print("####")
            # print(input_ids.shape)
            segments = (L - self.window_size) // self.inner_len
            if (L - self.window_size) % self.inner_len > self.edge_len:
                segments += 1
            elif segments == 0:
                segments += 1
            x = self.backbone(input_ids=input_ids[:, :self.window_size],
                              attention_mask=attention_mask[:, :self.window_size], return_dict=False)[0]
            for i in range(1, segments + 1):
                start = self.window_size - self.edge_len + (i - 1) * self.inner_len
                end = self.window_size - self.edge_len + (i - 1) * self.inner_len + self.window_size
                end = min(end, L)
                x_next = input_ids[:, start:end]
                mask_next = attention_mask[:, start:end]
                x_next = self.backbone(input_ids=x_next, attention_mask=mask_next, return_dict=False)[0]
                # L_next=x_next.shape[1]-self.edge_len,
                if i == segments:
                    x_next = x_next[:, self.edge_len:]
                else:
                    x_next = x_next[:, self.edge_len:self.edge_len + self.inner_len]
                # print(x_next.shape)
                x = torch.cat([x, x_next], 1)

                # print(start,end)
        # print(x.shape)
        x = self.lstm(x.permute(1, 0, 2))[0].permute(1, 0, 2)
        return x


class MultiScaleRegressionModule(BaseLightningModule):
    """This is the model module for regression"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None
        self.cls_pooling = CLSPooling()
        self.mean_pooling = MeanPooling()
        self.segment_backbone = SlidingWindowTransformerModel(self.config.model.path,
                                                              rnn="LSTM", window_size=256, edge_len=16)
        hidden_size = self.segment_backbone.backbone.config.hidden_size
        self.attention_pooling = AttentionPooling(hidden_size=hidden_size)
        self.head = nn.Linear(hidden_size * 2, 6)
        # self.seg_head = nn.Linear(hidden_size, 6)

    def compute_loss(self, logits, labels, weight=None):
        if self.has_weight:
            return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target], weights=weight)
        return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])

    def encode_doc_token(self, train_batch):
        x = train_batch[0]
        input_ids, attention_mask = x["input_ids"], x["attention_mask"]
        model_output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(model_output, "hidden_states"):
            hidden_states = model_output.hidden_states
        else:
            hidden_states = [model_output.last_hidden_state]
        doc_states = self.cls_pooling(hidden_states[-1])
        token_states = self.mean_pooling(hidden_states[-1], mask=x["attention_mask"])
        return doc_states + token_states

    def encode_seg(self, train_batch):
        x = train_batch[0]
        seg_states = self.segment_backbone(x["input_ids"], x["attention_mask"])
        return self.attention_pooling(seg_states, mask=x["attention_mask"])

    def forward(self, train_batch, un_batch=None):
        doc_token_states = self.encode_doc_token(train_batch)
        seg_states = self.encode_seg(train_batch)
        doc_token_score = self.head(torch.cat([seg_states, doc_token_states], dim=1))
        # seg_score = self.seg_head(seg_states)
        return doc_token_score

    def training_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)
        loss = self.compute_loss(logits, y, weight)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)
        loss = self.compute_loss(logits, y, weight=weight)
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        for i, idx in enumerate(self.trained_target):
            labels = np.array(self.val_metrics.targets)
            predictions = np.array(self.val_metrics.predictions)
            score = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
            self.log(f"val/{label_columns[idx]}", score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


class MultiScale2RegressionModule(BaseLightningModule):
    """This is the model module for regression"""

    def __init__(self, config: DictConfig):
        super().__init__(config)
        trained_target = self.config.train.trained_target
        if not trained_target:
            trained_target = [0, 1, 2, 3, 4, 5]
        self.trained_target = trained_target
        self.has_weight = "weight" in self.config.train.loss and self.config.train.reweight is not None
        hidden_size = self.backbone.config.hidden_size
        # label_columns = ["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]
        self.cohesion_pooling = MultiPooling("lstm_mean", hidden_size=hidden_size)
        self.cohesion_head = nn.Linear(hidden_size, 1)
        self.syntax_pooling = MultiPooling("lstm_mean", hidden_size=hidden_size)
        self.syntax_head = nn.Linear(hidden_size, 1)
        self.vocabulary_pooling = MultiPooling("weighted_mean", hidden_size=hidden_size, layer_start=4, num_hidden_layers=12)
        self.vocabulary_head = nn.Linear(hidden_size, 1)
        self.phraseology_pooling = MultiPooling("weighted_mean", hidden_size=hidden_size, layer_start=4, num_hidden_layers=12)
        self.phraseology_head = nn.Linear(hidden_size, 1)
        self.grammar_pooling = MultiPooling("lstm_mean", hidden_size=hidden_size)
        self.grammar_head = nn.Linear(hidden_size, 1)
        self.conventions_pooling = MultiPooling("lstm_mean", hidden_size=hidden_size)
        self.conventions_head = nn.Linear(hidden_size, 1)

    def compute_loss(self, logits, labels, weight=None):
        if self.has_weight:
            return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target], weights=weight)
        return self.criterion(logits[:, self.trained_target], labels[:, self.trained_target])

    def forward(self, train_batch, un_batch=None):
        x = train_batch[0]
        attention_mask = x["attention_mask"]
        feature = self.backbone(input_ids=x["input_ids"], attention_mask=attention_mask).hidden_states
        cohesion = self.cohesion_head(self.cohesion_pooling(feature[-1], mask=attention_mask))
        syntax = self.syntax_head(self.syntax_pooling(feature[-1], mask=attention_mask))
        vocab = self.vocabulary_head(self.vocabulary_pooling(feature, mask=attention_mask))
        phrase = self.phraseology_head(self.phraseology_pooling(feature, mask=attention_mask))
        grammar = self.grammar_head(self.grammar_pooling(feature[-1], mask=attention_mask))
        convention = self.conventions_head(self.conventions_pooling(feature[-1], mask=attention_mask))
        return torch.cat([cohesion, syntax, vocab, phrase, grammar, convention], dim=1)

    def training_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)
        loss = self.compute_loss(logits, y, weight)
        self.train_losses.update(loss.item(), n=y.shape[0])
        self.train_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                  logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("train/loss", self.train_losses.avg, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/score", self.train_metrics.score, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_index):
        if self.has_weight:
            x, y, weight = batch
        else:
            x, y = batch
            weight = None
        logits = self(batch)
        loss = self.compute_loss(logits, y, weight=weight)
        self.val_losses.update(loss.item(), n=y.shape[0])
        self.val_metrics.update(y.detach().cpu().numpy()[:, self.trained_target],
                                logits.detach().cpu().numpy()[:, self.trained_target])
        self.log("val/loss", self.val_losses.avg, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/score", self.val_metrics.score, prog_bar=True, on_step=False, on_epoch=True)
        for i, idx in enumerate(self.trained_target):
            labels = np.array(self.val_metrics.targets)
            predictions = np.array(self.val_metrics.predictions)
            score = mean_squared_error(labels[:, i], predictions[:, i], squared=False)
            self.log(f"val/{label_columns[idx]}", score, prog_bar=True, on_step=False, on_epoch=True)
        return loss


if __name__ == '__main__':
    from omegaconf import OmegaConf
    from transformers import AutoTokenizer

    model_path = "/media/heyao/42f00068-ba8d-48dd-8719-61eda5244b8d/pretrained-models/deberta-v3-base"
    model = SlidingWindowTransformerModel(model_path, window_size=128, edge_len=16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    text = "When a problem is a change you have to let it do the best on you no matter what is happening it can change your mind. sometimes you need to wake up and look what is around you because problems are the best way to change what you want to change along time ago. A\n\nproblem is a change for you because it can make you see different and help you to understand how tings wok.\n\nFirst of all it can make you see different then the others. For example i remember that when i came to the United States i think that nothing was going to change me because i think that nothing was going to change me because everything was different that my country and then i realist that wrong because a problem may change you but sometimes can not change the way it is, but i remember that i was really shy but i think that change a lot because sometimes my problems make me think that there is more thing that i never see in my life but i just need to see it from a different way and dont let nothing happened and ruing the change that i want to make because of just a problem. For example i think that nothing was going to change me and that i dont need to be shy anymore became i need to start seeing everything in a different ways because you can get mad at every one but you need to know what is going to happened after,\n\npeople may see you different but the only way that you know how to change is to do the best and don't let nothing or not body to change nothing about you. The way you want to change not one have that and can't do nothing about it because is your choice and your problems and you can decide what to do with it.\n\nsecond of all can help you to understand how things work. For instance my mom have a lot of problems but she have faith when she is around people, my mom is scare of high and i'm not scare of high i did not understand why my mos is scare of high and in not scare of high and every time i see my mom in a airplane it make me laugh because she is scare and is funny, but i see it from a different way and i like the high but also she have to understand that hoe things work in other people because it can no be the same as you. For example i think that my mom and me are different because we are and i have to understand that she does not like high and i need to understand that. to help someone to understand how things work you need to start to see how things work in that persons life.\n\nA problem is a change for you and can make you a different and help you to understand. Everyone has a different opinion and a different was to understand then others. everyone can see the different opinion and what other people think."
    encodings = tokenizer(text, return_tensors="pt")
    print(len(encodings["input_ids"][0]))
    output = model(encodings["input_ids"], encodings["attention_mask"])
    print(output.shape)
    config = OmegaConf.load("../../../config/deberta_v3_base_reg.yaml")
    model2 = MultiScaleRegressionModule(config)
    print(model2([encodings]))
