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
parser.add_argument('--ttt', dest='TTT', type=list, action='store', default=['ads', 1])
parser.add_argument('--tttt', dest='TTTT', type=dict, action='store', default={'adc': 1})

args = parser.parse_args()

print(ab.args_pack(args))
ab.args_log(args, TestArgBaseLogger)