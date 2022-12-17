"""
@created by: heyao
@created at: 2022-08-25 00:15:48
"""
from .regression.simple import FeedbackRegressionModule, FeedbackRegressionWithClassifierLossModule
from .regression.awp import AWPRegressionModule
from .regression.label_inject import LabelInjectRegressionModule, LabelInjectWithAvgRegressionModule
from .regression.one_target import FeedbackOneTargetRegressionModule
