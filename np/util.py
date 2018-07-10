# coding = utf-8

import numpy as np
from Putil.tf import util

_np_type = {
    32.0: np.float32,
    32: np.int32,
    64.0: np.float64,
    64: np.int64,
    8: np.uint8,
    -8: np.int8,
    16.0: np.float16,
    16: np.int16
}


class np_type:
    def __init__(self, label):
        self._type = _np_type[label]
        self._label = label
        pass

    @property
    def Type(self):
        return self._type

    def to_tf(self):
        return util._tf_type[self._label]
        pass
