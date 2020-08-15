# coding=utf-8

import argparse


def generate_args(parser):
    parser.add_argument('--auto_stop_patience', type=int, action='store', default=8, metavar='N', \
        help='when the time count to patience, it would stop, default: 8')
    parser.add_argument('--auto_stop_mode', type=str, action='store', default='max', \
        help='max or min, the change meaning the indicator is better, if the change do not match the mode, \
            we would cound the time, default: max')
    pass