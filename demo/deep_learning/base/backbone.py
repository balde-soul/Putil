# coding=utf-8
import torch
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
from Putil.demo.deep_learning.base.backbone_impl.backbone import Backbone, common_backbone_arg, VisionBackbone, DDBackbone
from Putil.demo.deep_learning.base.backbone_impl.resnet_cus import _resnet
from Putil.demo.deep_learning.base.backbone_impl.vgg_cus import _vgg


def vgg(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_vgg():
        return _vgg(args, property_type='', **kwargs)
    return generate_vgg


def vggArg(parser, property_type='', **kwargs):
    common_backbone_arg(parser, property_type='', **kwargs)
    parser.add_argument('--{}vgg_help'.format(property_type), action='store', type=str, default='', \
        help='vgg arch: [vgg11, vgg13, vgg16, vgg19, vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn] \n' \
            'supported downsample rate: []')
    pass

def resnet(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def generate_resent():
        return _resnet(temp_args, property_type, **kwargs)
    return generate_resent

def resnetArg(parser, property_type='', **kwargs):
    common_backbone_arg(parser, property_type, **kwargs)
    parser.add_argument('--{}resnet_help'.format(property_type), action='store', type=str, default='', \
        help='backbone_arch: [18, 34, 50, 101, 152, ext50_32x4d, ext101_32x8d, wide_50_2, wide_101_2] \n' \
            'backbone_pretrained: see the doc\n' \
                'backbone_weight_path: see the doc')
    parser.add_argument('--resnet_replace_stride_width_dilation', )
    pass


def unet(args, property_type='', **kwargs):
    pass


def unetArg(parser, property_type='', **kwargs):
    pass


class _DefaultBackbone(Backbone, Module):
    def __init__(self, args, property_type='', **kwargs):
        Backbone.__init__(self, args, property_type, **kwargs)
        Module.__init__(self)
        self._params = list()
        input_shape = 1
        for index, inter_cell in enumerate(eval('self._args.{}interlayer_cell'.format(property_type))):
            temp_param = torch.nn.Parameter(torch.rand([input_shape, inter_cell], dtype=torch.float32, requires_grad=True))
            self.register_parameter('inter_cell_{}'.format(index), temp_param)
            self._params.append((index, temp_param))
            input_shape = inter_cell
        pass

    def forward(self, x):
        out = x
        for (index, param) in self._params:
            out = torch.matmul(out, param)
        return out
    pass


def DefaultBackbone(args, property_type='', **kwargs):
    temp_args = copy.deepcopy(args)
    def _generate_default_backbone():
        return _DefaultBackbone(args, property_type, **kwargs)
    return _generate_default_backbone


def DefaultBackboneArg(parser, property_type='', **kwargs):
    common_backbone_arg(parser, property_type, **kwargs)
    parser.add_argument('--{}interlayer_cell'.format(property_type), nargs='+', type=int, default=[32, 32, 16], \
        help='the inter layer nerve cell amount, a list, the len represent the layer amount, the cell represent the nerve amount')
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