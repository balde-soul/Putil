# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('auto_stop_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.auto_stop as standard
from util import auto_stop as project
from importlib import reload
reload(standard)
reload(project)


def auto_stop_factory(args):
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
    model = '{0}.{1}'.format(args.auto_stop_source, args.auto_stop_name)
    logger.info('auto_stop model: {}, arch: {}'.format(model, args.auto_stop_arch))
    return eval('{}(args)'.format(model))
    

def auto_stop_arg_factory(parser, source, name):
    auto_stop_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('auto_stop_arg: {}'.format(auto_stop_arg))
    #import pdb; pdb.set_trace()
    return eval('{}(parser)'.format(auto_stop_arg)) 