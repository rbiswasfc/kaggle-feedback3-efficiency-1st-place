import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertAttention

#-------- Pooling ---------------------------------------------------------------#


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


#-------- Classification Head ---------------------------------------------------#


class BertAttentionFeedbackHead(nn.Module):
    """
    classification head with 
        - multi-head attention mechanism
        - weighted average of top transformer layers
    """

    def __init__(self, hidden_size, num_layers_in_head=6, num_targets=6):
        super(BertAttentionFeedbackHead, self).__init__()

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

        # apply attention
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]

        # pdb.set_trace()
        context_vector = self.pool(encoder_layer, attention_mask)  # mean pooling

        # compute logits
        logits = self.classifier(context_vector).reshape(-1, self.num_targets)

        return logits


class LstmBertAttentionFeedbackHead(nn.Module):
    """
    classification head with 
        - multi-head attention mechanism
        - weighted average of top transformer layers
    """

    def __init__(self, hidden_size, num_layers_in_head=6, num_targets=6):
        super(LstmBertAttentionFeedbackHead, self).__init__()

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

        # LSTM
        self.lstm_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Linear(hidden_size, num_targets)

    def forward(self, backbone_outputs, attention_mask):
        # weighted average of layers
        # pdb.set_trace()

        x = torch.stack(backbone_outputs.hidden_states[-self.num_layers_in_head:])
        w = F.softmax(self.weights, dim=0)
        encoder_layer = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)

        self.lstm_layer.flatten_parameters()
        encoder_layer = self.lstm_layer(encoder_layer)[0]

        # apply attention
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        encoder_layer = self.attention(encoder_layer, extended_attention_mask)[0]

        # pdb.set_trace()
        context_vector = self.pool(encoder_layer, attention_mask)  # mean pooling

        # compute logits
        logits = self.classifier(context_vector).reshape(-1, self.num_targets)

        return logits


class DepthLogitsFeedbackHead(nn.Module):
    """
    classification head with 
        - depth-wise residual flow
    """

    def __init__(self, hidden_size, num_layers_in_head=6, num_targets=6):
        super(DepthLogitsFeedbackHead, self).__init__()

        self.num_layers_in_head = num_layers_in_head
        self.num_targets = num_targets

        self.pool = MeanPooling()

        # learnable params weighted average of layers
        init_amax = 5
        weight_data = torch.linspace(-init_amax, init_amax, num_layers_in_head)
        weight_data = weight_data.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.weights = nn.Parameter(weight_data, requires_grad=True)

        # main classifier
        self.classifier = nn.Linear(hidden_size, num_targets)

    def forward(self, backbone_outputs, attention_mask):

        logits_list = []
        for i, depth_idx in enumerate(reversed(range(1, self.num_layers_in_head+1))):
            x = backbone_outputs.hidden_states[-depth_idx]  # (bs, max_len, h)
            x = self.pool(x, attention_mask)  # (bs, h)
            logits_i = self.classifier(x).reshape(-1, self.num_targets)
            logits_list.append(logits_i)

        # pdb.set_trace()
        enc = torch.stack(logits_list)  # (num_layers, bs, num_targets)
        w = F.softmax(self.weights, dim=0)
        logits = (w * enc).sum(dim=0)
        # .reshape(-1, self.num_targets)  # (bs, num_targets)

        return logits


class DepthResidualFeedbackHead(nn.Module):
    """
    classification head with 
        - depth-wise residual flow
    """

    def __init__(self, hidden_size, num_layers_in_head=6, num_targets=6):
        super(DepthResidualFeedbackHead, self).__init__()

        self.num_layers_in_head = num_layers_in_head
        self.num_targets = num_targets

        self.pool = MeanPooling()
        self.classifier = nn.Linear(hidden_size, num_targets)

    def forward(self, backbone_outputs, attention_mask):

        for i, depth_idx in enumerate(reversed(range(1, self.num_layers_in_head+1))):
            x = backbone_outputs.hidden_states[-depth_idx]  # (bs, max_len, h)
            x = self.pool(x, attention_mask)  # (bs, h)
            logits_i = self.classifier(x).reshape(-1, self.num_targets) / self.num_layers_in_head

            if i == 0:
                logits = logits_i
            else:
                logits = logits + logits_i  # residual

        return logits


class DepthLstmFeedbackHead(nn.Module):
    """
    classification head with 
        - depth-wise lstm flow
    """

    def __init__(self, hidden_size, num_layers_in_head=6, num_targets=6):
        super(DepthLstmFeedbackHead, self).__init__()

        self.num_layers_in_head = num_layers_in_head
        self.num_targets = num_targets

        self.pool = MeanPooling()
        self.classifier = nn.Linear(hidden_size, num_targets)

        # LSTM
        self.lstm_layer = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size//2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, backbone_outputs, attention_mask):

        encoding_list = []
        for i, depth_idx in enumerate(reversed(range(1, self.num_layers_in_head+1))):
            x = backbone_outputs.hidden_states[-depth_idx]  # (bs, max_len, h)
            x = self.pool(x, attention_mask)  # (bs, h)
            encoding_list.append(x)

        # stack
        lstm_input = torch.stack(encoding_list, dim=1)  # (bs, num_layers, h)

        self.lstm_layer.flatten_parameters()
        lstm_output = self.lstm_layer(lstm_input)[0]  # (bs, num_layers, h)

        # mean pool
        context_vector = torch.mean(lstm_output, dim=1)  # (bs, h)
        logits = self.classifier(context_vector).reshape(-1, self.num_targets)

        return logits


class TargetTokenFeedbackHead(nn.Module):
    """
    classification head with 
        - multi-head attention mechanism
        - context vector from target tokens
    """

    def __init__(self, hidden_size, num_layers_in_head=6, num_targets=6):
        super(TargetTokenFeedbackHead, self).__init__()

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
        self.classifier = nn.Linear(hidden_size, num_targets)

    def forward(self, backbone_outputs, target_token_idxs, target_pos):

        x = torch.stack(backbone_outputs.hidden_states[-self.num_layers_in_head:])
        w = F.softmax(self.weights, dim=0)
        encoder_layer = (w * x).sum(dim=0)  # (bs, max_len, hidden_size)

        # get context vectors of target token ids
        target_token_idxs = target_token_idxs[0]  # same for all examples in the batch
        encoder_layer = encoder_layer[:, target_token_idxs]

        # apply attention and mean pool
        encoder_layer = self.attention(encoder_layer)[0]  # (bs, 6, h)

        context_vector = encoder_layer[:, target_pos]  # (bs, h)
        # compute logits
        logits = self.classifier(context_vector).reshape(-1, self.num_targets)

        return logits
