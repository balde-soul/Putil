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
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('optimization implemented only in torch')
    optimization = '{}.{}'.format(args.optimization_source, args.optimization_name)
    return eval('{}(args)'.format(optimization))


def optimization_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('optimization_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 


from torch.optim import Optimizer

class Optimizations(Optimizer):
    pass