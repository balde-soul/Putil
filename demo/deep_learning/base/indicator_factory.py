# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('indicator_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.indicator as standard
from util import indicator as project


def indicator_factory(args, source, name, property_type='', **kwargs):
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
    model = '{0}.{1}'.format(source, name)
    return eval('{}(args, property_type, **kwargs)'.format(model))


##@brief
# @note
# @param[in] source
def indicator_arg_factory(parser, source, name, property_type='', **kwargs):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('indicator_arg: {}'.format(arg))
    return eval('{}(parser, property_type, **kwargs)'.format(arg)) 