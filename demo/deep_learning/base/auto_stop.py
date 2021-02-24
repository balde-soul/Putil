# coding=utf-8
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('auto_stop').logger()
logger.setLevel(plog.DEBUG)

from Putil.trainer.auto_stop import auto_stop as AutoStop
from Putil.trainer.auto_stop import AutoStop as _DefaultAutoStop


def common_auto_stop_arg(parser, property_type='', **kwargs):
    pass


def DefaultAutoStop(args, property_type='', **kwargs):
    '''
     @param[in] args
      args.auto_stop_patience
      args.auto_stop_mode
    '''
    def generate_default_auto_stop():
        return _DefaultAutoStop.generate_AutoStop_from_args(args)
    return generate_default_auto_stop


def DefaultAutoStopArg(parser, property_type='', **kwargs):
    _DefaultAutoStop.generate_args(parser, property_type)
