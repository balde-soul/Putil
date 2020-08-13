# coding=utf-8

import argparse


def generate_args(parser):
    parser.add_argument('--lr_reduce_init_lr', dest='LrReduceInitLr', type=float, action='store', default=0.1, \
        help='the init lr, defautl: 0.1')
    parser.add_argument('--lr_reduce_lr_factor', dest='LrReduceLrFactor', type=float, action='store', default=0.1, \
        help='the reduce factor for lr reducion, default: 0.1')
    parser.add_argument('--lr_reduce_lr_epsilon', dest='LrReduceLrEpsilon', type=float, action='store', default=0.001, \
        help='the threshold which decide the reducion is triggered or not, default: 0.001')
    parser.add_argument('--lr_reduce_lr_patience', dest='LrReduceLrPatience', type=int, action='store', default=3, \
        help='while reducion is triggered in LrReduceLrPatience time, the lr whould be reduce, default: 3')
    parser.add_argument('--lr_reduce_lr_cool_down', dest='LrReduceLrCoolDown', type=int, action='store', default=2, \
        help='the epoch in which the reducion trigger would not be count, default: 2')
    parser.add_argument('--lr_reduce_lr_min', dest='LrReduceLrMin', type=float, action='store', default=0.0000001, \
        help='the minimum of the lr, default: 0.0000001')
    parser.add_argument('--lr_reduce_mode', dest='LrReduceMode', type=str, action='store', default='max', \
        help='the mode which we want the indicator to be, \
            if not or the transformation is not upper the LrReduceEpsilon, \
                the reducion would be triggered, defautl: max')
    pass