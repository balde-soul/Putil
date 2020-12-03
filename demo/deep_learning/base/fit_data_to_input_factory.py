# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('fit_data_to_input_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.fit_data_to_input as standard
from ..util import fit_data_to_input as project


def fit_data_to_input_factory(args):
    '''
     @brief
     @note
     @param[in]
      args
      fit_data_to_input_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('fit_data_to_input of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.fit_data_to_input_source, args.fit_data_to_input_name)
    logger.info('fit_data_to_input model: {}, arch: {}'.format(model, args.fit_data_to_input_arch))
    return eval('{}(args)'.format(model))

def fit_data_to_input_arg_factory(parser, source, name):
    fit_data_to_input_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('fit_data_to_input_arg: {}'.format(name))
    return eval('{}(parser)'.format(fit_data_to_input_arg)) 
