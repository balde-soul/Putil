# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('model_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.model as standard
import util.model as project


def model_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       model_name: the main type of the model
       model_source: from standard or project
       model_arch: sub type in the model_name
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('model of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.model_source, args.model_name)
    logger.info('model model: {}, arch: {}'.format(model, args.model_arch))
    return eval('{}(args)'.format(model))
