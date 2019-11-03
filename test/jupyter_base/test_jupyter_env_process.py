# coding=utf-8

import sys
import os

print(sys.path)
import Putil.jupyter_base.jupyter_env_process as jep

jep.add_jupyter_notbook_path_to_env(os.path.join(os.getcwd(), 'test/jupyter_base/for_jupyter_import_test_code'))

print(sys.path)

import sub.for_jupyter_import_test as sfjit
sfjit.test()

import for_jupyter_import_test as fjit
fjit.test()
