# coding = utf-8

##@brief 生成常用的backbone参数
# @note generate the follow name space arg
# <property_type>backbone_arch：每个backbone类型可以分为几种核心架构，此参数定义生成的架构
# <property_type>backbone_downsample_rate: 一个backbone中会对输入数据进行下采样，此参数规定下采样尺寸
# <property_type>backbone_pretrained: 当该参数被set时，表示要加载backbone的预训练参数，同时backbone_weight_path必须要有相关的设置
# <property_type>backbone_weight_path：表示预训练模型参数文件的path，可以custom设置参数
def common_backbone_arg(parser, property_type='', **kwargs):
    parser.add_argument('--{}backbone_arch'.format(property_type), type=str, default='', action='store', \
        help='specify the arch of the backbone, such 19 for backbone_name with vgg')
    parser.add_argument('--{}backbone_downsample_rate'.format(property_type), type=int, default=None, action='store', \
        help='specify the downsample rate for the backbone')
    parser.add_argument('--{}backbone_pretrained'.format(property_type), default=False, action='store_true', \
        help='load the pretrained backbone weight or not')
    parser.add_argument('--{}backbone_weight_path'.format(property_type), type=str, default='', action='store', \
        help='specify the pre-trained model for the backbone, use while in finetune mode, '\
            'if the weight is specify, the backbone weight would be useless')
    pass


class Backbone:
    ##@brief
    # @param[in] args.backbone_pretrained
    def __init__(self, args, property_type='', **kwargs):
        self._args = args
        self._backbone_pretrained = eval('args.{}backbone_pretrained'.format(property_type))
        self._backbone_arch = eval('args.{}backbone_arch'.format(property_type))
        self._backbone_weight_path = eval('args.{}backbone_weight_path'.format(property_type))
        pass

    def get_backbone_pretrained(self):
        return self._backbone_pretrained
    backbone_pretrained = property(get_backbone_pretrained)

    def get_backbone_arch(self):
        return self._backbone_arch
    backbone_arch = property(get_backbone_arch)

    def get_backbone_weight_path(self):
        return self._backbone_weight_path
    backbone_weight_path = property(get_backbone_weight_path)
    pass


##@brief base common Backbone for 2D data
# @
class VisionBackbone(Backbone):
    ##@brief 
    # @param[in] args for the Backbone
    # @param[in] args.backbone_downsample_rate specify the downsample rate for the backbone
    def __init__(self, args, property_type='', **kwargs):
        Backbone.__init__(self, args, property_type, **kwargs)
        self._backbone_downsample_rate = eval('args.{}backbone_downsample_rate'.format(property_type))
        pass
    pass


##@brief base common Backbone for 2D data
# @
class DDBackbone(VisionBackbone):
    ##@brief 
    # @param[in] args for the Backbone
    # @param[in] args.backbone_downsample_rate specify the downsample rate for the backbone
    def __init__(self, args, property_type='', **kwargs):
        VisionBackbone.__init__(self, args, property_type, **kwargs)
        pass
    pass


class DDBackboneWithResolution(DDBackbone):
    def __init__(self, args, property_type='', **kwargs):
        DDBackbone.__init__(self, args, property_type, **kwargs)
        self._resolution_output = dict()
        pass

    def get_resolution_output(self):
        return self._resolution_output
    resolution_output = property(get_resolution_output)
