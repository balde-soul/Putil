# conding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('data_loader_factory').logger()
logger.setLevel(plog.DEBUG)
data_loader_factory_logger = logger.getChild('data_loader_factory')
data_loader_factory_logger.setLevel(plog.DEBUG)
from Putil.demo.deep_learning.base import data_loader as standard
from util import data_loader as project


def data_loader_factory(args, data_loader_source, data_loader_name, property_type='', **kwargs):
    '''
     @brief generate the callable obj
     @note 
     @param[in] args
      framework: 'torch'
    '''
    data_loader_factory_logger.info('use {} data loader'.format(args.framework))
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('data_loader of framework: {} is not implemented'.format(args.framework))
    data_loader = '{0}.{1}'.format(data_loader_source, data_loader_name)
    return eval('{}(args, property_type, **kwargs)'.format(data_loader))


def data_loader_arg_factory(parser, source, name, property_type='', **kwargs):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('data_loader_arg: {}'.format(arg))
    return eval('{}(parser, property_type, **kwargs)'.format(arg)) 