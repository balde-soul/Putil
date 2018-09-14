# coding=utf-8
from colorama import Fore


def information(step, information, format):
    base = ''
    if step == 0:
        base += '-' * 10
        base += information
        base += '-' * 10
    else:
        base += '>' * step
        base += information
    return format + base + Fore.RESET
