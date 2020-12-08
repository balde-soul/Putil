# coding=utf-8


def horovod(args):
    if args.framework == 'torch':
        import horovod.torch as hvd
        return hvd
    elif args.framework == 'tf':
        import horovod.tensorflow as hvd
        return hvd