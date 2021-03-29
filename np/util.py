# coding = utf-8

import numpy as np
from Putil.tf import util
import Putil.loger as plog
import sys
root_logger = plog.PutilLogConfig("NpUtil").logger()
root_logger.setLevel(plog.DEBUG)

NpTypeLogger = root_logger.getChild("NpTypeLogger")
NpTypeLogger.setLevel(plog.DEBUG)

_np_type = {
    0.32: np.float32,
    32: np.int32,
    0.64: np.float64,
    64: np.int64,
    8: np.uint8,
    -8: np.int8,
    0.16: np.float16,
    16: np.int16
}


class np_type:
    def __init__(self, label):
        try:
            self._type = _np_type[label]
        except KeyError as e:
            NpTypeLogger.error('key : {0} is not supported\n{1}'.format(label, e))
            sys.exit()
        self._label = label
        pass

    @property
    def Type(self):
        return self._type

    def to_tf(self):
        return util._tf_type[self._label]