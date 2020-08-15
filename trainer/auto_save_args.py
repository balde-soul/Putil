# coding=utf-8

import argparse


def generate_args(parser):
    parser.add_argument('--auto_save_mode', dest='auto_save_mode', type=str, action='store', default='max', help='the AutoSaverImprove, default: True')
    parser.add_argument('--auto_save_delta', dest='auto_save_delta', type=float, action='store', default=0.001, help='the AutoSaverDelta, default: 0.001')
    parser.add_argument('--auto_save_keep_save_range', dest='auto_save_keep_save_range', type=list, action='store', default=[], help='the AutoSaverKeepSaveRange, default: []')
    parser.add_argument('--auto_save_abandon_range', dest='auto_save_abandon_range', type=list, action='store', default=[], help='the AutoSaverAbandonRange, default: []')
    parser.add_argument('--auto_save_base_line', dest='auto_save_base_line', type=int, action='store', default=None, help='the AutoSaverBaseLine, default: None')
    parser.add_argument('--auto_save_limit_line', dest='auto_save_limit_line', type=int, action='store', default=None, help='the AutoSaverLimitLine, default: None')
    parser.add_argument('--auto_save_history_amount', dest='auto_save_history_amount', type=int, action='store', default=100, help='the AutoSaverHistoryAmount, default: 100')
    pass