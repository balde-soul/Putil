# coding=utf-8
from colorama import Fore
from enum import Enum
import copy
from torch.nn import Module
from abc import ABCMeta, abstractmethod
from Putil.base import logger as plog

root_logger = plog.PutilLogConfig('backbone').logger()
root_logger.setLevel(plog.DEBUG)
BackboneLogger = root_logger.getChild('Backbone')
BackboneLogger.setLevel(plog.DEBUG)
DefaultBackboneLogger = root_logger.getChild('DefaultBackbone')
DefaultBackboneLogger.setLevel(plog.DEBUG)
from Putil.torch.pretrained_model.vgg import VGG


def common_backbone_arg(parser):
    '''
     @brief 生成常用的backbone参数
     @note
      backbone_arch：每个backbone类型可以分为几种核心架构，此参数定义生成的架构
      backbone_downsample_rate: 一个backbone中会对输入数据进行下采样，此参数规定下采样尺寸
      backbone_pretrained: 当该参数被set时，表示要加载backbone的预训练参数，同时backbone_weight_path必须要有相关的设置
      backbone_weight_path：表示预训练模型参数文件的path，可以custom设置参数
    '''
    parser.add_argument('--backbone_arch', type=str, default='', action='store', \
        help='specify the arch of the backbone, such 19 for backbone_name with vgg')
    parser.add_argument('--backbone_downsample_rate', type=int, default=None, action='store', \
        help='specify the downsample rate for the backbone')
    parser.add_argument('--backbone_pretrained', default=False, action='store_true', \
        help='load the pretrained backbone weight or not')
    parser.add_argument('--backbone_weight_path', type=str, default='', action='store', \
        help='specify the pre-trained model for the backbone, use while in finetune mode, '\
            'if the weight is specify, the backbone weight would be useless')
    pass


class Backbone:
    def __init__(self, args):
        self._backbone_pretrained = args.backbone_pretrained
        self._backbone_name = args.backbone_name
        self._backbone_arch = args.backbone_arch
        self._backbone_weight_path = args.backbone_weight_path
        self._backbone_downsample_rate = args.backbone_downsample_rate
        pass
    pass


class _vgg(Backbone, Module):
    def __init__(self, args):
        Backbone.__init__(self, args)
        Module.__init__(self)
        self._vgg = VGG(self._backbone_arch, self._backbone_downsample_rate, self._backbone_weight_path, self._backbone_pretrained)
    
    def forward(self, x):
        return self._vgg(x)
    pass


def vgg(args):
    temp_args = copy.deepcopy(args)
    def generate_vgg():
        return _vgg(args)
    return generate_vgg


def vggArg(parser):
    common_backbone_arg(parser)
    pass


from Putil.torch.pretrained_model.resnet import _ResNet
from torchvision.models.resnet import model_urls
from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.utils import load_state_dict_from_url
resnet_block = {'18': BasicBlock, '34': BasicBlock, '50': Bottleneck, \
    '101': Bottleneck, '152': Bottleneck, 'ext50_32x4d': Bottleneck, 'ext101_32x8d': Bottleneck, \
        'wide_50_2': Bottleneck, 'wide_101_2': Bottleneck}
resnet_layer = {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3], '50': [3, 4, 6, 3], \
    '101': [3, 4, 23, 3], '152': [3, 8, 36, 3], 'ext50_32x4d': [3, 4, 6, 3], 'ext101_32x8d': [3, 4, 23, 3], \
        'wide_50_2': [3, 4, 6, 3], 'wide_101_2': [3, 4, 23, 3]}
groups = {'18': None, '34': None, '50': None, \
    '101': None, '152': None, 'ext50_32x4d': 32, 'ext101_32x8d': 32, \
        'wide_50_2': None, 'wide_101_2': None}
width_per_group_type = {'18': None, '34': None, '50': None, \
    '101': None, '152': None, 'ext50_32x4d': 4, 'ext101_32x8d': 8, \
        'wide_50_2': 64 * 2, 'wide_101_2': 64 * 2}
def _resnet(arch, block, layers, pretrained, progress, model_dir, **kwargs):
    model = _ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress, 
                                              model_dir=model_dir)
        model.load_state_dict(state_dict)
    return model

class _resnet(Backbone, Module):
    def __init__(self, args):
        Backbone.__init__(self, args)
        Module.__init__(self)
        param = dict()
        if self._backbone_arch != 'custom':
            if groups[self._backbone_arch] is not None:
                param['groups'] = groups[self._backbone_arch]
                pass
            if width_per_group_type[self._backbone_arch] is not None:
                param['width_per_group_type'] = width_per_group_type[self._backbone_arch]
                pass
            self._backbone = _resnet(self._backbone_arch, resnet_block[self._backbone_arch], \
                layers=resnet_layer[self._backbone_arch][0: self._backbone_downsample_rate], \
                pretrained=self._backbone_pretrained, progress=True, model_dir=self._backbone_weight_path, **param)
            pass
        else:
            raise NotImplementedError('custom resnet is not implemented')
            pass
        pass

    def forward(self, x):
        pass

def resnet(args):
    temp_args = copy.deepcopy(args)
    def generate_resent():
        return _resnet(temp_args)
    return generate_resent

def resnetArg(parser):
    common_backbone_arg(parser)
    parser.add_argument('--resnet_help', action='store', type=str, default='', \
        help='backbone_arch: [18, 34, 50, 101, 152, ext50_32x4d, ext101_32x8d, wide_50_2, wide_101_2] \n' \
            'backbone_pretrained: see the doc\n' \
                'backbone_weight_path: see the doc')
    parser.add_argument('--resnet_replace_stride_width_dilation', )
    pass


def unet(args):
    pass


def unetArg(parser):
    pass


class DefaultBackbone(Backbone, Module):
    def __init__(self, args):
        Backbone.__init__(self, args)
        Module.__init__(self)
        pass

    def forward(self, x):
        return x
    pass


def DefaultBackboneArg(parser):
    common_backbone_arg(parser)
    pass
#
#
#def a(**kwargs):
#    print(kwargs)
#    pass
#
#def b(t, **kwargs):
#    a(**kwargs)
#
#k = dict()
#k['a'] = 1
#k['b'] = 1
#b(1, c=1, **k)