# coding=utf-8
import sys

print(sys.path)
import Putil.jupyter_base.jupyter_env_process as jep

jep.add_jupyter_notbook_path_to_env('/home/sins/Download/')

print(sys.path)

import Cloud.cloud2 as cloud
