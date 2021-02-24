# coding=utf-8

import argparse


def generate_args(parser, property_type=''):
    parser.add_argument('--{}auto_save_mode'.format(property_type), type=str, action='store', default='max', help='the AutoSaverImprove, default: True')
    parser.add_argument('--{}auto_save_delta'.format(property_type), type=float, action='store', default=0.001, help='the AutoSaverDelta, default: 0.001')
    parser.add_argument('--{}auto_save_keep_save_range'.format(property_type), type=list, action='store', default=[], help='the AutoSaverKeepSaveRange, default: []')
    parser.add_argument('--{}auto_save_abandon_range'.format(property_type), type=list, action='store', default=[], help='the AutoSaverAbandonRange, default: []')
    parser.add_argument('--{}auto_save_base_line'.format(property_type), type=int, action='store', default=None, help='the AutoSaverBaseLine, default: None')
    parser.add_argument('--{}auto_save_limit_line'.format(property_type), type=int, action='store', default=None, help='the AutoSaverLimitLine, default: None')
    parser.add_argument('--{}auto_save_history_amount'.format(property_type), type=int, action='store', default=100, help='the AutoSaverHistoryAmount, default: 100')
    pass