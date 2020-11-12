# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('backbone_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.backbone as standard
import util.backbone as project


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