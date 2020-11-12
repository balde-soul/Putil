# coding=utf-8
from Putil.demo.deep_learning.base.backbone import Backbone

class DefaultBackbone(Backbone):
    def __init__(self):
        Backbone.__init__(self, args)

    def __call__(self, *input):
        raise NotImplementedError('DefaultBackbone is not implemented')
    pass
