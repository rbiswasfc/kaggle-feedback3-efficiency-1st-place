"""
@created by: heyao
@created at: 2022-06-23 13:02:06
"""
import torch
import torch.nn as nn

from feedback_ell.nn.heads import GAU
from feedback_ell.nn.poolers.meanmax import MeanPooling
from feedback_ell.nn.poolers.base import BasePooling


class AttentionPooling(BasePooling):
    def __init__(self, *, hidden_size, **kwargs):
        super(AttentionPooling, self).__init__()

        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, mask=None):
        w = self.attention(x).float()
        if mask is not None:
            w[mask == 0] = float('-inf')
        w = torch.softmax(w, 1)
        x = torch.sum(w * x, dim=1)
        return x


class GAUPooling(BasePooling):
    def __init__(self, *, hidden_size, **kwargs):
        super(GAUPooling, self).__init__()
        self.gau = GAU(dim=hidden_size)

    def forward(self, x, mask=None):
        return self.gau(x)


class AttentionHead(nn.Module):
    def __init__(self, *, hidden_size):
        super().__init__()
        self.in_features = hidden_size
        self.middle_features = hidden_size

        self.W = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)
        self.out_features = hidden_size

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True):
        super(Attention, self).__init__()
        self.supports_masking = True
        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0
        weight = torch.zeros(feature_dim, 1)
        nn.init.xavier_uniform_(weight)
        self.weight = nn.Parameter(weight)
        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim
        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(-1, step_dim)
        if self.bias:
            eij = eij + self.b
        eij = torch.tanh(eij)
        a = torch.exp(eij)
        if mask is not None:
            a = a * mask
        a = a / torch.sum(a, 1, keepdim=True) + 1e-10
        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class AttentionHeadMeanPooling(BasePooling):
    def __init__(self, *, hidden_size, **kwargs):
        super().__init__()
        self.attention_pool = AttentionHead(hidden_size=hidden_size)
        self.mean_pool = MeanPooling()

    def forward(self, x, mask=None):
        attn = self.attention_pool(x, mask=mask)
        mean = self.mean_pool(x, mask=mask)
        return torch.cat([attn, mean], dim=1)


if __name__ == '__main__':
    x = torch.rand((4, 128, 768))
    pooling = GAUPooling(hidden_size=768)
    print(pooling(x).shape)
