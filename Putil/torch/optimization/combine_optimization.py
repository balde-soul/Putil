# coding=utf-8
#from torch.optim import Optimizer
import torch


class CombineOptimization:
    def __init__(self, **optimizations):
        self._optimizations = optimizations
        pass

    def step(self, closure=None):
        for index, (k, v) in enumerate(self._optimizations.items()):
            v.step()
            pass
        pass

    def load_state_dict(self, state_dict, unexisted_strategy):
        for index, (k, v) in enumerate(self._optimizations.items()):
            if k in state_dict.dict():
                v.load_state_dict(state_dict[k])
                pass
            else:
                pass
            pass
        pass