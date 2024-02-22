# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('indicator_statistic_factory').logger()
logger.setLevel(plog.DEBUG)
IndicatorStatisticFactoryLogger = logger.getChild('indicator_statistic_factory')
IndicatorStatisticFactoryLogger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.indicator_statistic as standard
from util import indicator_statistic as project


def indicator_statistic_factory(args, source, name, property_type='', **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
       indicator_statistic_name: the main type of the indicator_statistic
       indicator_statistic_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('indicator_statistic of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(source, name)
    IndicatorStatisticFactoryLogger.info('indicator_statistic_factory: {}|{}'.format(model, property_type))
    return eval('{}(args, property_type, **kwargs)'.format(model))


def indicator_statistic_arg_factory(parser, source, name, property_type='', **kwargs):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('indicator_statistic_arg: {}|{}'.format(arg, property_type))
    return eval('{}(parser)'.format(arg)) 