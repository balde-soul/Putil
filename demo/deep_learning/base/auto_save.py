# coding=utf-8
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('auto_save').logger()
logger.setLevel(plog.DEBUG)

from Putil.trainer.auto_save import auto_save as AutoSave
from Putil.trainer.auto_save import AutoSave as _DefaultAutoSave


class DefaultAutoSave:
    def __init__(self):
        pass
    
    def __call__(self, args):
        pass

def DefaultAutoSave(args):
    '''
     @param[in] args
      args.auto_save_mode
      args.auto_save_delta
      args.auto_save_keep_save_range
      args.auto_save_abandon_range
      args.auto_save_base_line
      args.auto_save_limit_line
      args.auto_save_history_amount
    '''
    def generate_default_auto_save():
        return _DefaultAutoSave.generate_AutoSave_from_args(args)
    return generate_default_auto_save


def DefaultAutoSaveArg(parser):
    _DefaultAutoSave.generate_args(parser)