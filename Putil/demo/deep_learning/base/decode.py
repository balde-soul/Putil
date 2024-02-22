# coding=utf-8
import copy
from abc import abstractmethod
from torch.nn import Module


##@brief
# @note
class Decode:
    '''
     @brief
     @note 解码模型输出，生成直接通用结果
    '''
    def __init__(self, args, property_type='', **kwargs):
        self._fit_to_decode_input = kwargs.get('fit_to_indicator_input', None)
        pass

    ##@brief decode the data
    # @note 当output为None的时候，表示通过datas进行解析
    # @param[in] datas dataset产生的datas
    # @param[in] output model的输出
    def __call__(self, datas, output=None):
        kargs = self._fit_to_decode_input(datas, output) if self._fit_to_decode_input is not None else (datas, output)
        return self._call_impl(*kargs)

    @abstractmethod
    def _call_impl(self, *kargs, **kwargs):
        pass
    pass


def CenterNetDecode(args):
    pass


def CenterNetDecodeArg(parser, property_type='', **kwargs):
    parser.add_argument('--{}center_net_decode_threshold'.format(property_type), type=float, default=0.1, action='store', \
        )
    pass


class _DefaultDecode(Decode, Module):
    def __init__(self, args, property_type='', **kwargs):
        Decode.__init__(self, args, property_type, **kwargs)
        Module.__init__(self)
        pass

    def _call_impl(self, *kargs, **kwargs):
        if kargs[1] is None:
            return kargs[0]
        else:
            return kargs[1]
        pass
    pass

def DefaultDecode(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_default_decode():
        return _DefaultDecode(args)
    return generate_default_decode

def DefaultDecodeArg(parser, property_type='', **kwargs):
    pass