# coding=utf-8
'''
 @note 使用backbone 与 backend进行model的构建，model包含了一个模型所有需要存储的参数
'''
from torch.nn import Module


class Model:
    def __init__(self, args):
        self._model_name = args.model_name
        self._model_source = args.model_source
        pass
    pass


class DefaultModel(Model):
    def __init__(self, args):
        Model.__init__(self, args)
        pass
    pass


def DefaultModel(parser):
    pass