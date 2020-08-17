# coding=utf-8
import Putil.base.logger as plog
plog.set_internal_debug_log()
import Putil.trainer.auto_stop as aust
import argparse

parser = argparse.ArgumentParser()

aust.AutoStop.generate_args(parser)
args = parser.parse_args()
auto_stop = aust.AutoStop.generate_AutoStop_from_args(args)
auto_stop.info()

for i in range(0, auto_stop.Patience * 2):
    assert auto_stop.stop_or_not(i) is False
    pass

a = list(range(0, auto_stop.Patience * 2))
a.reverse()
checkpoint_index = auto_stop.Patience - 3
state_dict = None

for index, i in enumerate(a):
    stop = auto_stop.stop_or_not(i)
    assert stop is (False if i > auto_stop.Patience else True), print(i)
    if stop:
        pass
    else:
        print('go on ')
        pass
    if index == checkpoint_index:
        print('save')
        state_dict = auto_stop.state_dict()
    pass
pass

auto_stop.load_state_dict(state_dict)
print('load')
for index, i in enumerate(a[checkpoint_index + 1:]):
    stop = auto_stop.stop_or_not(i)
    assert stop is (False if i > auto_stop.Patience else True), print(i)
    if stop:
        pass
    else:
        print('go on ')
    pass
pass