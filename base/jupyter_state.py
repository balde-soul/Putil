# coding=utf-8

import os

has_go_to_top = False

def go_to_top(num, now_path):
    if "JPY_PARENT_PID" in os.environ:
        global has_go_to_top
        if not has_go_to_top:
            for _ in range(0, num):
                now_path = os.path.dirname(now_path)
            os.chdir(now_path)
            has_go_to_top = True
            pass
        pass
    else:
        None
    pass