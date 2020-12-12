# coding=utf-8
import Putil.base.logger as plog
logger = plog.PutilLogConfig('optimization_factory').logger()
logger.setLevel(plog.DEBUG)
from Putil.demo.deep_learning.base import optimization as standard
from util import optimization as project
from importlib import reload
reload(standard)
reload(project)


def optimization_factory(args):
    '''
     @note generate
    '''
    if args.optimization_name == '':
        pass
    else:
        raise NotImplementedError('auto save: {} is not implemented'.format(args.optimization_name))
    optimization = '{}.{}'.format(args.optimization_source, args.optimization_name)
    return eval('{}(args)'.format(optimization))


def optimization_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('optimization_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 