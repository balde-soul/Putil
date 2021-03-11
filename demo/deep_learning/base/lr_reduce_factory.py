# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('lr_reduce_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.lr_reduce as standard
from util import lr_reduce as project


def lr_reduce_factory(args, source, name, property_type='', **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
      lr_reduce_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('lr_reduce of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(source, name)
    logger.info('lr_reduce: {}'.format(model))
    return eval('{}(args, property_type, **kwargs)'.format(model))


def lr_reduce_arg_factory(parser, source, name, property_type='', **kwargs):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('lr_reduce_arg: {}'.format(arg))
    return eval('{}(parser, property_type, **kwargs)'.format(arg))



