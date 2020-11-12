# coding=utf-8
from torch.nn import Module
from abc import ABCMeta, abstractmethod
from Putil.torch.pretrained_model.vgg import VGG


class Backbone:
    def __init__(self, args):
        self._backbone_pretrained = args.backbone_pretrained
        self._backbone_name = args.backbone_name
        self._backbone_arch = args.backbone_arch
        self._backbone_weight_path = args.backbone_weight_path
        self._backbone_downsample_rate = args.backbone_downsample_rate
        assert self._backbone_downsample_rate is not None
        pass


class vgg(Backbone, Module):
    def __init__(self, args):
        Backbone.__init__(args)
        Module.__init__(self)
        self._vgg = VGG(self._backbone_arch, self._backbone_downsample_rate, self._backbone_weight_path, self._backbone_pretrained)
    
    def forward(self, x):
        return self._vgg(x)