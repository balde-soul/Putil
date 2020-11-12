# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('indicator_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.indicators as standard
import util.indicator as project


def indicator_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       indicator_name: the main type of the indicator
       indicator_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('indicator of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.indicator_source, args.indicator_name)
    return eval('{}(args)'.format(model))
