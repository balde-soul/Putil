# coding=utf-8
import torch.nn.functional as F
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from collections import OrderedDict
from colorama.ansi import Back
from Putil.base import logger as plog

logger = plog.PutilLogConfig('mobilenet_cus').logger()
logger.setLevel(plog.DEBUG)
MobileNetLogger = logger.getChild('MobileNet')
MobileNetLogger.setLevel(plog.DEBUG)
import Putil.torch.pretrained_model.mobilenet2 as v2
import Putil.torch.pretrained_model.mobilenet1 as v1
from Putil.demo.deep_learning.base.backbone_impl.backbone import DDBackboneWithResolution


class MobileNetCusv1(DDBackboneWithResolution, v1.MobileNet):
    def __init__(self, args, property_type='', **kwargs):
        DDBackboneWithResolution.__init__(self, args, property_type, **kwargs)
        v1.MobileNet.__init__(self)
        pass

    def forward(self, x):
        #x = self.features(x)
        temp_shape = x.shape
        MobileNetLogger.debug(x.shape)
        for index, feature in enumerate(self.model):
            x = feature(x)
            if sum([0 if old == new else 1 for index, (old, new) in enumerate(zip(x.shape[2:], temp_shape[2:]))]) > 0:
                self._resolution_output[len(self._resolution_output.items())] = x
                temp_shape = x.shape
            MobileNetLogger.debug('{}: {}'.format(index, x.shape))
            pass
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x
    pass

##@brief
# @note
class MobileNetCusv2(DDBackboneWithResolution, v2.MobileNetV2):
    def __init__(self, args, property_type='', **kwargs):
        DDBackboneWithResolution.__init__(self, args, property_type, **kwargs)
        v2.MobileNetV2.__init__(self)
        pass

    def forward(self, x):
        #x = self.features(x)
        temp_shape = x.shape
        MobileNetLogger.debug(x.shape)
        for index, feature in enumerate(self.features):
            x = feature(x)
            if sum([0 if old == new else 1 for index, (old, new) in enumerate(zip(x.shape[2:], temp_shape[2:]))]) > 0:
                self._resolution_output[len(self._resolution_output.items())] = x
                temp_shape = x.shape
            MobileNetLogger.debug('{}: {}'.format(index, x.shape))
            pass
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), x.size(1))
        x = self.classifier(x)
        return x
    pass


def mobilenet_v2_args(parser, property_type, **kwargs):
    pass


##@brief
# @note
# @param[in]
# @param[in]
# @return 
def _mobilenet(args, property_type='', **kwargs):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    if eval('args.{}backbone_arch'.format(property_type)) == 'v1':
        Model = MobileNetCusv1
    if eval('args.{}backbone_arch'.format(property_type)) == 'v2':
        Model = MobileNetCusv2
        model_urls = v2.model_urls
    model = Model(args, property_type, **kwargs)
    if model.backbone_pretrained and eval('args.{}backbone_arch'.format(property_type)) == 'v2':
        state_dict = load_state_dict_from_url(
            model_urls['mobilenet_v2'],
            model_dir=eval('args.{}backbone_weight_path'.format(property_type)),
        )
        _pretrained_dict = OrderedDict()
        for idx, (k, v) in enumerate(state_dict.items()):
            splitted_k = k.split('.')

            # 0-5, 306-311
            if idx in list(range(0, 6)):
                splitted_k.insert(2, 'conv')

            if idx in list(range(306, 312)):
                splitted_k.insert(2, 'conv')

            if 'classifier' in splitted_k:
                splitted_k[1] = '0'

            _pretrained_dict['.'.join(splitted_k)] = v

        model.load_state_dict(_pretrained_dict)
    return model