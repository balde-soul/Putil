# coding=utf-8
from Putil.demo.deep_learning.base.evaluate_indicator import EvaluateIndicator


class DefaultEvaluateIndicator(EvaluateIndicator):
    def __init__(self):
        EvaluateIndicator.__init__(self, args)

    def __call__(self, *input):
        raise NotImplementedError('DefaultEvaluateIndicator is not implemented')
    pass