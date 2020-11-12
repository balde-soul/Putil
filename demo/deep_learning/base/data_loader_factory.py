# conding=utf-8
import Putil.base.logger as plog
from Putil.demo.deep_learning.base.data_loader import *

logger = plog.PutilLogConfig('data_loader_factory').logger()
logger.setLevel(plog.DEBUG)
data_loader_factory_logger = logger.getChild('data_loader_factory')
data_loader_factory_logger.setLevel(plog.DEBUG)


def data_loader_factory(args):
    data_loader_factory_logger.info('use {} data loader'.format(args.framework))
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('data_loader of framework: {} is not implemented'.format(args.framework))
    return eval('{}(args)'.format('{}_{}'.format(args.framework, DataLoader)))
    pass