# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('fit_decode_to_result_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.fit_decode_to_result as standard
from ..util import fit_decode_to_result as project


def fit_decode_to_result_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
      fit_decode_to_result_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('fit_decode_to_result of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.fit_decode_to_result_source, args.fit_decode_to_result_name)
    logger.info('fit_decode_to_result model: {}, arch: {}'.format(model, args.fit_decode_to_result_arch))
    return eval('{}(args)'.format(model))

def fit_decode_to_result_arg_factory(parser, source, name):
    fit_decode_to_result_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('fit_decode_to_result_arg: {}'.format(name))
    return eval('{}(parser)'.format(fit_decode_to_result_arg)) 
