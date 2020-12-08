# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('backbone_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.backbone as standard
from util import backbone as project


def backbone_factory(args):
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
    model = '{0}.{1}'.format(args.backbone_source, args.backbone_name)
    logger.info('backbone model: {}, arch: {}'.format(model, args.backbone_arch))
    return eval('{}(args)'.format(model))
    

def backbone_arg_factory(parser, source, name):
    backbone_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('backbone_arg: {}'.format(backbone_arg))
    return eval('{}(parser)'.format(backbone_arg)) 