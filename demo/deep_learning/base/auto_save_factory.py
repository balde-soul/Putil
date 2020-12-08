# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('auto_save_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.auto_save as standard
from util import auto_save as project


def auto_save_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
      auto_save_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('auto_save of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.auto_save_source, args.auto_save_name)
    logger.info('auto_save model: {}, arch: {}'.format(model, args.auto_save_arch))
    return eval('{}(args)'.format(model))

def auto_save_arg_factory(parser, source, name):
    auto_save_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('auto_save_arg: {}'.format(name))
    return eval('{}(parser)'.format(auto_save_arg)) 