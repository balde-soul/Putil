# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('backend_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.backend as standard
from util import backend as project
from importlib import reload
reload(standard)
reload(project)


def backend_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
      backend_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('backend of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.backend_source, args.backend_name)
    logger.info('backend model: {}, arch: {}'.format(model, args.backend_arch))
    return eval('{}(args)'.format(model))


def backend_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('backend_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 