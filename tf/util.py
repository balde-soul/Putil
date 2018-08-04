# coding = utf-8
import tensorflow as tf
from Putil.np import util
import Putil.loger as plog
import sys

root_logger = plog.PutilLogConfig("TfUtil").logger()
root_logger.setLevel(plog.DEBUG)

TfTypeLogger = root_logger.getChild("TfTypeLogger")
TfTypeLogger.setLevel(plog.DEBUG)

_tf_type = {
    0.32: tf.float32,
    32: tf.int32,
    0.64: tf.float64,
    64: tf.int64,
    8: tf.uint8,
    -8: tf.int8,
    0.16: tf.float16,
    16: tf.int16
}


class tf_type:
    def __init__(self, label):
        try:
            self._type = _tf_type[label]
        except KeyError as e:
            TfTypeLogger.error('key: {0} is not supported\n{1}'.formay(label, e))
            sys.exit()
        self._label = label
        pass

    @property
    def Type(self):
        return self._type

    def to_np(self):
        return util._np_type[self._label]

    @staticmethod
    def to_np(tf_dtype):
        for _reflect in _tf_type.items():
            if _reflect[1] == tf_dtype:
                return util._np_type[_reflect[0]]
            else:
                pass
            pass
        pass
