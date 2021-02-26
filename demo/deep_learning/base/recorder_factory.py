# coding=utf-8
import Putil.demo.deep_learning.base.recorder as standard
import util.recorder as project


def recorder_factory(args):
    return eval('{}.{}(args)'.format(args.recorder_source, args.recorder_name))


def recorder_arg_factory(parser, source, name):
    eval('{}.{}Arg(parser)'.format(source, name))
    pass