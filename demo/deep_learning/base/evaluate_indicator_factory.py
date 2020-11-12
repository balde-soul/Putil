# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('evaluate_indicator_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.evaluate_indicator as standard
import util.evaluate_indicator as project


def evaluate_indicator_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       evaluate_indicator_name: the main type of the evaluate_indicator
       evaluate_indicator_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('evaluate_indicator of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.evaluate_indicator_source, args.evaluate_indicator_name)
    return eval('{}(args)'.format(model))

