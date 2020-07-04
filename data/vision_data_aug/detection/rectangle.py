# coding=utf-8

import Putil.data.aug as pAug


class RectAngleFunc(pAug.AugFunc):
    def __init__(self, func):
        pAug.AugFunc.__init__(self)
        self._func = func
        pass

    def _generate_doc(self):
        return ''

    def _generate_name(self):
        return ''
    pass


def RectAngleFuncGenerator(config):
    def func(config):
        def _func(*args):
            return args
        return _func
    return [RectAngleFunc(_func_) for _func_ in [func(config)]]