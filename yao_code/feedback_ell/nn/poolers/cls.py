"""
@created by: heyao
@created at: 2022-09-06 13:55:38
"""
from feedback_ell.nn.poolers import BasePooling


class CLSPooling(BasePooling):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x, mask=None):
        return x[:, 0, :]
