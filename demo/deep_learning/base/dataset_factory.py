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
    def _combine(source, name, target):
        target = '{} {}.{}'.format(target, source, name)
        return True
    dataset = ''
    [_combine(source, name, target) for source, name in zip(args.dataset_sources, args.dataset_names)]
    logger.info('augs: {}'.format(target))
    return [eval('{}.{}(args)'.format(aug_source, aug_name)) for aug_source, aug_name in zip(args.aug_sources, args.aug_names)]


def dataset_arg_factory(parser, source, name):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('dataset_arg: {}'.format(arg))
    return eval('{}(parser)'.format(arg)) 