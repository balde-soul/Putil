'''
 @note
 本文件提供了原始数据到模型输入数据的编码方法，实现了CommonData中ConvertToInput
'''
# coding=utf-8
import Putil.data.convert_to_input as IOConvertor
IOConvertor = IOConvertor.IOConvertor


class Encode(IOConvertor):
    def __init__(self):
        IOConvertor.__init__(self)
        pass

    def __call__(self, *args):
        raise NotImplementedError('not Implemented')
    pass