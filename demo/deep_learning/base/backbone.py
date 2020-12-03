# coding=utf-8
from torch.nn import Module
from abc import ABCMeta, abstractmethod
from Putil.torch.pretrained_model.vgg import VGG


def common_backbone_arg(parser):
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
        assert self._backbone_downsample_rate is not None


class vgg(Backbone, Module):
    def __init__(self, args):
        Backbone.__init__(self, args)
        Module.__init__(self)
        self._vgg = VGG(self._backbone_arch, self._backbone_downsample_rate, self._backbone_weight_path, self._backbone_pretrained)
    
    def forward(self, x):
        return self._vgg(x)
    pass


def vggArg(parser):
    pass


class DefaultBackbone(Backbone, Module):
    def __init__(self, args):
        Backbone.__init__(self, args)
        Module.__init__(self)

    def forward(self, x):
        return x
    pass


def DefaultBackboneArg(parser):
    common_backbone_arg(parser)
    pass