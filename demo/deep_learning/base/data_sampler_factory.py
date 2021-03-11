# conding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('data_sampler_factory').logger()
logger.setLevel(plog.DEBUG)

from Putil.demo.deep_learning.base import data_sampler as standard
from util import data_sampler as project


def data_sampler_factory(args, data_sampler_source, data_sampler_name, property_type='', **kwargs):
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('data_loader of framework: {} is not implemented'.format(args.framework))
    data_sampler = '{}.{}'.format(data_sampler_source, data_sampler_name)
    return eval('{}(args, property_type, **kwargs)'.format(data_sampler))


def data_sampler_arg_factory(parser, source, name, property_type='', **kwargs):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('data_sampler_arg: {}'.format(arg))
    return eval('{}(parser, property_type, **kwargs)'.format(arg)) 
