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

for i in indicator:
    auto_save.save_or_not(i)
    pass