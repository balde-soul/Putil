# coding=utf-8
from enum import Enum
from Putil.base import logger as plog

logger = plog.PutilLogConfig('putil_status').logger()
logger.setLevel(plog.DEBUG)

##@brief PutilMode
# @note 代表着Putil库的运行等级，不同等级影响了不同的行为，比如log，有些log是在PutilMode.Debug中才起效的
class PutilMode(Enum):
    # Debug等级，在debug代码时使用
    Debug = 0
    # Release等级，在正常运行中使用
    Release = 1
    pass

putil_mode = PutilMode.Release

def set_putil_mode(mode):
    global putil_mode
    putil_mode = mode
    pass

def putil_is_debug():
    global putil_mode
    return putil_mode == PutilMode.Debug

def putil_is_release():
    return putil_mode == PutilMode.Release