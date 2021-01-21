# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('data_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.dataset as standard
from util import dataset as project
from importlib import reload
reload(standard)
reload(project)


def dataset_factory(args, property_type='', **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
       data_name: the main type of the data
       data_source: from standard or project
    '''
    model = '{}.{}'.format(args.dataset_sources[property_type], args.dataset_names[property_type])
    logger.info('dataset: {}'.format(model))
    return eval('{}(args, property_type, **kwargs)'.format(model))


def dataset_arg_factory(parser, source, name, property_type='', **kwargs):
    arg = '{}.{}Arg'.format(source, name)
    logger.info('dataset_arg: {}'.format(arg))
    return eval('{}(parser, property_type, **kwargs)'.format(arg)) 