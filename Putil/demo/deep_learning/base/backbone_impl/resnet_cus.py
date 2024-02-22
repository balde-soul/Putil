# coding=utf-8
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from torchvision.models.resnet import BasicBlock
from torchvision.models.resnet import Bottleneck
from torchvision.models.utils import load_state_dict_from_url
# 18:resnet18 Deep Residual Learning for Image Recognition
# ```https://arxiv.org/pdf/1512.03385.pdf```
# 34:resnet34 Deep Residual Learning for Image Recognition
# ```https://arxiv.org/pdf/1512.03385.pdf```
# 50:resnet50 Deep Residual Learning for Image Recognition
# ```https://arxiv.org/pdf/1512.03385.pdf```
# 101:resnet101 Deep Residual Learning for Image Recognition 
# ```https://arxiv.org/pdf/1512.03385.pdf>```
# 152:resnet152 Deep Residual Learning for Image Recognition
# ```https://arxiv.org/pdf/1512.03385.pdf>```
# ext50_32x4d:resnext50_32x4d model from Aggregated Residual Transformation for Deep Neural Networks
# ```https://arxiv.org/pdf/1611.05431.pdf```
# ext101_32x8d:resnext101_32x8d Aggregated Residual Transformation for Deep Neural Networks
# ```https://arxiv.org/pdf/1611.05431.pdf>```
# wide_50_2:wide_resnet50_2 Wide Residual Networks
# ```https://arxiv.org/pdf/1605.07146.pdf```
# wide_101_2:wide_resnet101_2 Wide Residual Networks
# ```https://arxiv.org/pdf/1605.07146.pdf```

model_urls = {
    '18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    '152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'ext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'ext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}
resnet_block = {'18': BasicBlock, '34': BasicBlock, '50': Bottleneck, \
    '101': Bottleneck, '152': Bottleneck, 'ext50_32x4d': Bottleneck, 'ext101_32x8d': Bottleneck, \
        'wide_50_2': Bottleneck, 'wide_101_2': Bottleneck}
resnet_layer = {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3], '50': [3, 4, 6, 3], \
    '101': [3, 4, 23, 3], '152': [3, 8, 36, 3], 'ext50_32x4d': [3, 4, 6, 3], 'ext101_32x8d': [3, 4, 23, 3], \
        'wide_50_2': [3, 4, 6, 3], 'wide_101_2': [3, 4, 23, 3]}
groups = {'default': 1, '18': 1, '34': 1, '50': 1, \
    '101': 1, '152': 1, 'ext50_32x4d': 32, 'ext101_32x8d': 32, \
        'wide_50_2': 1, 'wide_101_2': 1}
width_per_group_type = {'default': 64, '18': 64, '34': 64, '50': 64, \
    '101': 64, '152': 64, 'ext50_32x4d': 4, 'ext101_32x8d': 8, \
        'wide_50_2': 64 * 2, 'wide_101_2': 64 * 2}

from Putil.demo.deep_learning.base.backbone_impl.backbone import DDBackboneWithResolution
import Putil.torch.pretrained_model.resnet as resnet

class ResnetCus(resnet.ResNet, DDBackboneWithResolution):
    def __init__(self, args, property_type='', **kwargs):
        DDBackboneWithResolution.__init__(self, args, property_type, **kwargs)
        self._resnet_replace_stride_width_dilation = eval('args.{}resnet_replace_stride_width_dilation'.format(property_type))
        resnet.ResNet.__init__(self, resnet_block[self._backbone_arch],
            layers=resnet_layer[self._backbone_arch], 
            groups=groups[self._backbone_arch if self._backbone_arch is not None else 'default'],
            width_per_group=width_per_group_type[self._backbone_arch if self._backbone_arch is not None else 'default'],
            replace_stride_with_dilation=self._resnet_replace_stride_width_dilation)
        pass

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        self._resolution_output[1] = x
        if self._backbone_downsample_rate >= 2:
            x = self.layer1(x)
            self._resolution_output[2] = x
        if self._backbone_downsample_rate >= 3:
            x = self.layer2(x)
            self._resolution_output[3] = x
        if self._backbone_downsample_rate >= 4:
            x = self.layer3(x)
            self._resolution_output[4] = x
        if self._backbone_downsample_rate >= 5:
            x = self.layer4(x)
            self._resolution_output[5] = x
        x = F.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    pass

def _resnet(args, property_type, **kwargs):
    model = ResnetCus(args, property_type, **kwargs)
    if model.backbone_pretrained:
        state_dict = load_state_dict_from_url(model_urls[model.backbone_arch], 
        eval('args.{}backbone_weight_path'.format(property_type)))
        model.load_state_dict(state_dict)
        # release
        del state_dict
    return model