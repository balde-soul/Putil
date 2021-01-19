# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('backbone_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.backbone as standard
from util import backbone as project
from importlib import reload
reload(standard)
reload(project)


def backbone_factory(args, property_type='', **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
       backbone_name: the main type of the backbone
       backbone_source: from standard or project
       backbone_arch: sub type in the backbone_name
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('backbone of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}|{2}'.format(args.backbone_source, args.backbone_name, property_type)
    logger.info('backbone model: {}, arch: {}'.format(model, args.backbone_arch))
    return eval('{}(args, property_type, **kwargs)'.format(model))
    

def backbone_arg_factory(parser, source, name, property_type='', **kwargs):
    backbone_arg = '{0}.{1}Arg|{2}'.format(source, name, property_type)
    logger.info('backbone_arg: {}'.format(backbone_arg))
    return eval('{}(parser, property_type, **kwargs)'.format(backbone_arg)) 