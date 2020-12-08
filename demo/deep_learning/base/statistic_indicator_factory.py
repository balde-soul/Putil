# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('statistic_indicator_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.statistic_indicator as standard
from util import statistic_indicator as project


def statistic_indicator_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       statistic_indicator_name: the main type of the statistic_indicator
       statistic_indicator_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('statistic_indicator of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.statistic_indicator_source, args.statistic_indicator_name)
    return eval('{}(args)'.format(model))


def statistic_indicator_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('statistic_indicator_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 