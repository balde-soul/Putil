from torch.nn import Module


from Putil.demo.deep_learning.base.backbone_impl.backbone import DDBackboneWithResolution
from Putil.torch.pretrained_model.vgg import VGG

##@brief the VGG backbone
# @note
class _vgg(DDBackboneWithResolution, VGG):
    ##@brief waiting for completing
    # @param[in] args.
    def __init__(self, args, property_type='', **kwargs):
        DDBackboneWithResolution.__init__(self, args, property_type, **kwargs)
        VGG.__init__(self, self._backbone_arch, self._backbone_downsample_rate, \
            self._backbone_weight_path, self._backbone_pretrained)
        pass
    
    def forward(self, x):
        return VGG.forward(self, x)
    pass
