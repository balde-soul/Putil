# coding=utf-8
from importlib import reload
import copy
from Putil.demo.deep_learning.base import accumulated_opt as standard
from util import accumulated_opt as project
reload(standard)
reload(project)


def accumulated_opt_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_accumulated_opt():
        #print(temp_args)
        return eval('{}.{}_{}(temp_args)'.format(temp_args.accumulated_opt_source, \
            temp_args.framework, temp_args.accumulated_opt_name))
    if args.framework == 'torch':
        return generate_accumulated_opt
    else:
        raise NotImplementedError('{}{} is not implemented in  framework {}'.format(args.accumulated_opt_source, args.accumulated_opt_name, args.framework))
    pass

def accumulated_opt_arg_factory(parser, source, name):
    eval('{}.{}Arg(parser)'.format(source, name))
    pass