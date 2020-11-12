# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('loss_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.losss as standard
import util.loss as project


def loss_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       loss_name: the main type of the loss
       loss_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('loss of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.loss_source, args.loss_name)
    return eval('{}(args)'.format(model))

