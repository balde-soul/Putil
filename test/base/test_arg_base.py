# coding=utf-8
import argparse
import Putil.base.arg_base as ab
import Putil.base.logger as plog

plog.set_internal_debug_log()

TestArgBaseLogger = plog.PutilLogConfig('test_arg_base').logger()
TestArgBaseLogger.setLevel(plog.DEBUG)

parser = argparse.ArgumentParser()
parser.add_argument('--t', dest='T', type=str, action='store', default='asdas')
parser.add_argument('--tt', dest='TT', type=str, action='store', default='asd')

args = parser.parse_args()

print(ab.args_pack(args))
ab.args_log(args, TestArgBaseLogger)