"""
@created by: heyao
@created at: 2022-09-17 16:46:30
"""
import torch
import torch.nn as nn
from transformers.activations import ACT2FN


class DebertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class DebertaOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lm_head = DebertaLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.lm_head(sequence_output)
        return prediction_scores
