# conding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('data_sampler_factory').logger()
logger.setLevel(plog.DEBUG)

from Putil.demo.deep_learning.base import data_sampler as standard
from util import data_sampler as project
from importlib import reload
reload(standard)
reload(project)


def data_sampler_factory(args):
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('data_loader of framework: {} is not implemented'.format(args.framework))
    data_sampler = '{}.{}'.format(args.data_sampler_source, args.data_sampler_name)
    return eval('{}(args)'.format(data_sampler))


def data_sampler_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('data_sampler_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 
