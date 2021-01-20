# coding=utf-8


def horovod(args):
    if args.framework == 'torch':
        import horovod.torch as hvd
        return hvd
    elif args.framework == 'tf':
        import horovod.tensorflow as hvd
        return hvd
    pass


def horovod_arg(parser):
    parser.add_argument('--hvd_reduce_mode', type=str, action='store', default='Average', \
        help='the reduce mode for horovod, supports Average,Sum and AdaSum')
    parser.add_argument('--hvd_compression_mode', type=str, action='store', default='none', \
        help='the compression mode for horovod, supports none, mro and fp16')
    pass


def horovod_reduce_mode_is_adasum(args):
    return args.hvd_reduce_mode == 'AdaSum'


def horovod_reduce_mode_is_average(args):
    return args.hvd_reduce_mode == 'Average'

def horovode_reduce_mode_is_sum(args):
    return args.hvd_reduce_mode == 'Sum'