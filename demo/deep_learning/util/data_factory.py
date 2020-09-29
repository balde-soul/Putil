# coding=utf-8
#from uitl.data import *


def data_factory(args):
    if args.data_name == '':
        raise NotImplementedError('data_factory: {} is not implemented'.format(args.data_name))
    return eval('(args)'.format(args.data_name))
    pass
#
#
#class t:
#    def __init__(self, args):
#        pass
#
#    def __call__(self):
#        print('t')
#    pass
#
#
#args = None
#a = eval('t(args)')
#a()
#print(eval.__doc__)