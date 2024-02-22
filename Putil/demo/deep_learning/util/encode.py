'''
 @note
 本文件提供了原始数据到模型输入数据的编码方法，实现了CommonData中ConvertToInput
'''
# coding=utf-8
from Putil.demo.deep_learning.base.encode import Encode


class DefaultEncode(Encode):
    def __init__(self):
        Encode.__init__(self, args)

    def __call__(self, *input):
        raise NotImplementedError('DefaultEncode is not implemented')
    pass