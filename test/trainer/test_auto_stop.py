# coding=utf-8
import Putil.trainer.auto_stop as aust
import argparse

parser = argparse.ArgumentParser()

aust.AutoStop.generate_args(parser)
args = parser.parse_args()
auto_stop = aust.AutoStop.generate_AutoStop_from_args(args)

for i in range(0, auto_stop.Patience * 2):
    assert auto_stop.stop_or_not(i) is False
    pass
a = list(range(0, auto_stop.Patience * 2))
a.reverse()
for i in a:
    assert auto_stop.stop_or_not(i) is False if i > auto_stop.Patience else True, print(i)
    pass
pass