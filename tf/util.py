# coding = utf-8
import tensorflow as tf

tf_type = {
    32.0: tf.float32,
    32: tf.int32,
    64.0: tf.float64,
    64: tf.int64,
    8: tf.uint8,
    -8: tf.int8,
    16.0: tf.float16,
    16: tf.int16
}
