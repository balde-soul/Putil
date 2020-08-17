# coding=utf-8
import Putil.base.logger as plog
plog.set_internal_debug_log()

import Putil.trainer.lr_reduce as lr
import argparse
parser = argparse.ArgumentParser()

lr.LrReduce.generate_args(parser)
args = parser.parse_args()
lr_reduce = lr.LrReduce.generate_LrReduce_from_args(args)

lr_reduce.info()

indicator = [0, -1, 0.001, 0.001,  0.001, 0.001, 0.001, 0.001, 0.1, 0, 0, 0, 0, 0]

checkpoint_index = 6
state_dict = None

for index, i in enumerate(indicator):
    lr_reduce.reduce_or_not(i)
    if index == checkpoint_index:
        state_dict = lr_reduce.state_dict()
        print('save')

print('load')
lr_reduce.load_state_dict(state_dict)
for index, i in enumerate(indicator[checkpoint_index + 1:]):
    lr_reduce.reduce_or_not(i)
