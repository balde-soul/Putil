# coding=utf-8
from abc import abstractmethod, ABCMeta
import Putil.base.logger as plog


logger = plog.PutilLogConfig('lr_reduce').logger()
logger.setLevel(plog.DEBUG)

from Putil.trainer.lr_reduce import lr_reduce as LrReduce
from Putil.trainer.lr_reduce import LrReduce as _DefaultLrReduce

def DefaultLrReduce(args):
    '''
     @param[in] args
      args.lr_reduce_init_lr
      args.lr_reduce_lr_factor
      args.lr_reduce_lr_epsilon
      args.lr_reduce_lr_patience
      args.lr_reduce_cool_down
      args.lr_reduce_lr_min
      args.lr_reduce_mode
    '''
    return _DefaultLrReduce.generate_LrReduce_from_args(args)