# coding=utf-8
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('auto_save').logger()
logger.setLevel(plog.DEBUG)

from Putil.trainer.auto_save import auto_save as AutoSave
from Putil.trainer.auto_save import AutoSave as _DefaultAutoSave

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
    return _DefaultAutoSave.generate_AutoSave_from_args(args)