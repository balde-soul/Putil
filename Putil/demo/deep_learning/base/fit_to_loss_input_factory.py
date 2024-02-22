# coding=utf-8
import Putil.base.logger as plog

logger = plog.PutilLogConfig('fit_to_loss_input_factory').logger()
logger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.fit_to_loss_input as standard
from util import fit_to_loss_input as project


def fit_to_loss_input_factory(args, property_type='', **kwargs):
    '''
     @brief
     @note
     @param[in]
      args
      fit_to_loss_input_name: 0?_main name_1?1_sub type
      0?_: customized or ''
      1?_: pretrained or ''
    '''
    if args.framework == 'torch':
        pass
    else:
        raise NotImplementedError('fit_to_loss_input of framework: {} is not implemented'.format(args.framework))
    model = '{0}.{1}'.format(args.fit_to_loss_input_sources[property_type], args.fit_to_loss_input_names[property_type])
    logger.info('fit_to_loss_input: {}'.format(model))
    return eval('{}(args, property_type, **kwargs)'.format(model))

def fit_to_loss_input_arg_factory(parser, source, name, property_type='', **kwargs):
    fit_to_loss_input_arg = '{0}.{1}Arg'.format(source, name)
    logger.info('fit_to_loss_input_arg: {}'.format(name))
    return eval('{}(parser, property_type, **kwargs)'.format(fit_to_loss_input_arg)) 
