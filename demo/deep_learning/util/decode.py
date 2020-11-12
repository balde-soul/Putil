'''
 @note
 本文件提供了模型输出到常规输出的转换，用于train evaluate test中，解码模型的输出到常规的表示，便于对接
 可视化，以及其他的计算
'''
# coding=utf-8
from Putil.demo.deep_learning.base.decode import Decode


class DefaultDecode(Decode):
    def __init__(self):
        Decode.__init__(self, args)

    def __call__(self, *input):
        raise NotImplementedError('DefaultDecode is not implemented')
    pass