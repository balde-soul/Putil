# coding=utf-8
import copy
from abc import abstractmethod
from torch.nn import Module


class Decode:
    '''
     @brief
     @note 解码模型输出，生成直接通用结果
    '''
    def __init__(self, args):
        self._decode_name = args.decode_name
        self._decode_source = args.decode_source
    
    #@abstractmethod
    #def forward(self, *input, **kwargs):
    #    '''
    #     @brief
    #     @note 
    #     @param[in] *input
    #      两种模式:
    #      gt模式：解析Dataset的输出，生成通用结果，只需要输入Dataset的输出datas
    #      pre模式：解析模型的输出，生成通用结果，需要输入Dataset的输出datas与model的输出pre
    #     @param[in] **kwargs
    #      summary：bool 如果为True，通用结果会进行summary，如果为False，则不会进行summary
    #      step：当进行summary时，需要指定step
    #     @ret output tuple
    #    '''
    #    pass

    #@abstractmethod
    #def output_reflect(self):
    #    '''
    #     @note 输出通用结果的名称映射，key为output tuple索引，value为其对应名称
    #     @ret reflect
    #    '''
    #    pass
    pass


def CenterNetDecode(args):
    pass


def CenterNetDecodeArg(parser):
    parser.add_argument('--center_net_decode_threshold', type=float, default=0.1, action='store', \
        )
    pass


class _DefaultDecode(Decode, Module):
    def __init__(self, args):
        Decode.__init__(self, args)
        Module.__init__(self)
        pass
    pass

def DefaltDecode(args):
    temp_args = copy.deepcopy(args)
    def generate_default_decode():
        return _DefaultDecode(args)
    return generate_default_decode

def DefaultDecodeArg(parser):
    pass