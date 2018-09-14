# coding=utf-8

from optparse import OptionParser
import Putil.loger as plog
parser = OptionParser(usage='usage %prog [options] arg1 arg2')
level_default = 'Debug'
parser.add_option(
    '--level',
    action='store',
    dest='Level',
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
parser.add_option(
    '--test_mrsa',
    action='store_true',
    default=False,
    dest='TestMRSA',
    help='set this flag while you want to test MRSA'
)
parser.add_option(
    '--test_zero',
    action='store_true',
    default=False,
    dest='TestZero',
    help='set this flag while you want to test Zero'
)
parser.add_option(
    '--test_xavier',
    action='store_true',
    default=False,
    dest='TestXavier',
    help='set this flag while you want to test Xavier'
)
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")
from Putil.tf.param import InitializeParam
from Putil.tf.param import Initialize
import tensorflow as tf
from colorama import Fore

root_logger = plog.PutilLogConfig("tf/test/test_static").logger()
root_logger.setLevel(plog.DEBUG)
TestMrsaLogger = root_logger.getChild("TestMrsa")
TestMrsaLogger.setLevel(plog.DEBUG)
TestXavierLogger = root_logger.getChild("TestXavier")
TestXavierLogger.setLevel(plog.DEBUG)
TestZeroLogger = root_logger.getChild("TestZero")
TestZeroLogger.setLevel(plog.DEBUG)


def __test_mrsa():
    tf.reset_default_graph()
    param = {'method': 'mrsa'}
    param_fix = InitializeParam(param, default=param['method']).ShowDefault().complement(type=0.64).fix_with_default().ParamGenWithInfo(TestMrsaLogger)
    init = Initialize(param_fix).Initializer
    weight = tf.get_variable('weight', shape=[10], initializer=init)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pass


def __test_xavier():
    tf.reset_default_graph()
    param = {'method': 'xavier'}
    param_fix = InitializeParam(param, default=param['method']).ShowDefault().complement(type=0.64).fix_with_default().ParamGenWithInfo(TestXavierLogger)
    init = Initialize(param_fix).Initializer
    weight = tf.get_variable('weight', shape=[10], initializer=init)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    pass


def __test_zero():
    tf.reset_default_graph()
    param = {'method': 'zeros'}
    param_fix = InitializeParam(param, default=param['method']).ShowDefault().complement(type=0.64).fix_with_default().ParamGenWithInfo(TestZeroLogger)
    init = Initialize(param_fix).Initializer
    weight = tf.get_variable('weight', shape=[10], initializer=init)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())


if __name__ == '__main__':
    if options.TestMRSA:
        __test_mrsa()
        pass
    if options.TestXavier:
        __test_xavier()
        pass
    if options.TestZero:
        __test_zero()
        pass
    else:
        pass
    pass


