# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('data_type_adapter_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.data_type_adapter as standard
from util import data_type_adapter as project


def data_type_adapter_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
      data_type_adapter_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('data_type_adapter of framework: {} is not implemented'.format(args.framework))
    data_type_adapter = '{0}.{1}'.format(args.data_type_adapter_source, args.data_type_adapter_name)
    logger.info('data_type_adapter data_type_adapter: {}, arch: {}'.format(data_type_adapter, args.data_type_adapter_arch))
    return eval('{}(args)'.format(data_type_adapter))

def data_type_adapter_arg_factory(parser, source, name):
    data_type_adapter_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('data_type_adapter_arg: {}'.format(name))
    return eval('{}(parser)'.format(data_type_adapter_arg)) 
