# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('aug_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.aug as standard
from util import aug as project


def aug_factory(args, property_type='', **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
      aug_source: the aug source follow the format source_one-source_two-source-three-...
      aug_name: the aug name follow the format aug_one-aug_two-aug_three-...
    '''
    model = '{}.{}'.format(args.aug_sources[property_type], args.aug_names[property_type])
    logger.info('aug: {}|{}'.format(model, property_type))
    return eval('{}(args, property_type)'.format(model))

def aug_arg_factory(parser, source, name, property_type='', **kwargs):
    #import pdb; pdb.set_trace()
    model = '{}.{}Arg'.format(source, name)
    logger.info('aug_arg: {}|{}'.format(model, property_type))
    eval('{}(parser, property_type)'.format(model))