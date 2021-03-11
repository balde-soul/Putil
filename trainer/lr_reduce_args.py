# coding=utf-8

import argparse


def generate_args(parser, property_type=''):
    parser.add_argument('--{}lr_reduce_init_lr'.format(property_type), type=float, action='store', default=0.1, \
        help='the init lr, defautl: 0.1')
    parser.add_argument('--{}lr_reduce_lr_factor'.format(property_type), type=float, action='store', default=0.1, \
        help='the reduce factor for lr reducion, default: 0.1')
    parser.add_argument('--{}lr_reduce_lr_epsilon'.format(property_type), type=float, action='store', default=0.001, \
        help='the threshold which decide the reducion is triggered or not, default: 0.001')
    parser.add_argument('--{}lr_reduce_lr_patience'.format(property_type), type=int, action='store', default=3, \
        help='while reducion is triggered in lr_reduce_lr_patience time, the lr whould be reduce, default: 3')
    parser.add_argument('--{}lr_reduce_lr_cool_down'.format(property_type), type=int, action='store', default=2, \
        help='the epoch in which the reducion trigger would not be count, default: 2')
    parser.add_argument('--{}lr_reduce_lr_min'.format(property_type), type=float, action='store', default=0.0000001, \
        help='the minimum of the lr, default: 0.0000001')
    parser.add_argument('--{}lr_reduce_mode'.format(property_type), type=str, action='store', default='max', \
        help='the mode which we want the indicator to be, \
            if not or the transformation is not upper the lr_reduce_lr_patience, \
                the reducion would be triggered, defautl: max')
    pass