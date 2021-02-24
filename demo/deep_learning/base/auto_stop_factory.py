# coding=utf-8
from importlib import reload
import Putil.base.logger as plog

logger = plog.PutilLogConfig('auto_stop_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.auto_stop as standard
from util import auto_stop as project
reload(standard)
reload(project)


def auto_stop_factory(args, source, name, property_type='', **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
      auto_stop_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('auto_stop of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(source, name)
    logger.info('auto_stop: {}|{}'.format(model, property_type))
    return eval('{}(args, property_type, **kwargs)'.format(model))
    

def auto_stop_arg_factory(parser, source, name, property_type='', **kwargs):
    auto_stop_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('auto_stop_arg: {}|{}'.format(auto_stop_arg, property_type))
    #import pdb; pdb.set_trace()
    return eval('{}(parser, property_type, **kwargs)'.format(auto_stop_arg))