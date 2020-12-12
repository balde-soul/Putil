# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('encode_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.encode as standard
from util import encode as project
from importlib import reload
reload(standard)
reload(project)


def encode_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       encode_name: the main type of the encode
       encode_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('encode of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.encode_source, args.encode_name)
    return eval('{}(args)'.format(model))


def encode_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('encode_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 