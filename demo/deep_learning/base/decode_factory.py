# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('decode_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.decode as standard
from util import decode as project


def decode_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       decode_name: the main type of the decode
       decode_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('decode of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.decode_source, args.decode_name)
    return eval('{}(args)'.format(model))


def decode_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('decode_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 