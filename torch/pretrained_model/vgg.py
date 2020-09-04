# coding=utf-8
from enum import Enum
from torchviz.dot import make_dot

import torch
from torch import nn
from torch.nn import Module
from torchvision.models import vgg
from torchvision.models.utils import load_state_dict_from_url
from torch.autograd import Variable


def make_layers(cfg, downsample, batch_norm=False):
    resolution_output = []
    layers = []
    in_channels = 3
    downsample_time = 0
    final_cfg = list()
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            resolution_output.append(layers[-1])
            downsample_time += 1
            if downsample == 2 ** downsample_time:
                final_cfg.append(v)
                break
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            pass
        final_cfg.append(v)
        pass
    seq = nn.Sequential(*layers)
    return seq, resolution_output, final_cfg


class VGG(Module):
    class VGGArch(Enum):
        vgg11 = 'vgg11'
        vgg13 = 'vgg13'
        vgg16 = 'vgg16'
        vgg19 = 'vgg19'
        vgg11_bn = 'vgg11_bn'
        vgg13_bn = 'vgg13_bn' 
        vgg16_bn = 'vgg16_bn' 
        vgg19_bn = 'vgg19_bn' 
    vgg_arch_url_dic = {
        VGGArch.vgg11.name: {'cfg': vgg.cfgs['A'], 'url': vgg.model_urls['vgg11']},
        VGGArch.vgg13.name: {'cfg': vgg.cfgs['B'], 'url': vgg.model_urls['vgg13']},
        VGGArch.vgg16.name: {'cfg': vgg.cfgs['D'], 'url': vgg.model_urls['vgg16']},
        VGGArch.vgg19.name: {'cfg': vgg.cfgs['E'], 'url': vgg.model_urls['vgg19']},
        VGGArch.vgg11_bn.name: {'cfg': vgg.cfgs['A'], 'url': vgg.model_urls['vgg11_bn']},
        VGGArch.vgg13_bn.name: {'cfg': vgg.cfgs['B'], 'url': vgg.model_urls['vgg13_bn']},
        VGGArch.vgg16_bn.name: {'cfg': vgg.cfgs['D'], 'url': vgg.model_urls['vgg16_bn']},
        VGGArch.vgg19_bn.name: {'cfg': vgg.cfgs['E'], 'url': vgg.model_urls['vgg19_bn']}
    }
    def __init__(self, vgg_arch, downsample, model_dir, load_pretrained):
        Module.__init__(self)
        self.features, self._resolution_output, self._final_cfg = make_layers(
            VGG.vgg_arch_url_dic[vgg_arch]['cfg'], downsample)
        if load_pretrained:
            state_dict = load_state_dict_from_url(VGG.vgg_arch_url_dic[vgg_arch]['url'], progress=True, model_dir=model_dir)
            self.load_state_dict(state_dict, strict=False)
    
    def forward(self, x):
        return self.features(x)

    @property
    def resolution_output(self):
        return self._resolution_output

    @property
    def final_cfg(self):
        return self._final_cfg
    pass


#model = VGG(VGG.VGGArch.vgg11, 4, './', False)
#piter = model.named_parameters()
#while True:
#    try:
#        p = piter.__next__()
#        print(model.named_parameters().__next__()[0])
#    except Exception as e:
#        print(e)
#        break
#    pass
#print(model.resolution_output)