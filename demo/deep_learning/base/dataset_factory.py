# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('data_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.dataset as standard
from util import dataset as project


def dataset_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
       data_name: the main type of the data
       data_source: from standard or project
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('data of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.data_source, args.data_name)
    return eval('{}(args)'.format(model))


def dataset_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('dataset_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 