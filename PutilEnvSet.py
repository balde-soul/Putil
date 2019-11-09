# coding=utf-8
import os
import Putil.jupyter_base.jupyter_env_process as jep
jep.add_jupyter_notbook_path_to_env(os.path.dirname(os.path.abspath(__name__)))