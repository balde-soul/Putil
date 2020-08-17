# coding=utf-8
import Putil.base.logger as plog
plog.set_internal_debug_log()
import Putil.trainer.auto_save as aus
import argparse

parser = argparse.ArgumentParser()

aus.AutoSave.generate_args(parser)
args = parser.parse_args()

auto_save = aus.AutoSave.generate_AutoSave_from_args(args)
auto_save.info()

indicator = [0, -1, 0.0001, 0.1, 0.2]

state_dict = None
reload_index = 3
for index, i in enumerate(indicator):
    auto_save.save_or_not(i)
    if index == reload_index:
        print('checkpoint')
        state_dict = auto_save.state_dict()
    pass

print('load')
auto_save.load_state_dict(state_dict)
for i in indicator[reload_index + 1: ]:
    auto_save.save_or_not(i)
    pass
    