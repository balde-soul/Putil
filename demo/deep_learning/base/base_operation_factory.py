# coding=utf-8
import copy
import Putil.demo.deep_learning.base.base_operation as standard
import util.base_operation as project


def checkpoint_factory(args):
    pass


def save_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_save_func():
        if temp_args.framework == 'torch':
            return standard.torch_save
        else:
            raise NotImplementedError('not implemented')
        pass
    return generate_save_func


def deploy_factory(args):
    pass


def get_models_factory(args):
    pass


def load_saved_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_load_saveed_func():
        return eval('{}.{}_load_saved'.format(temp_args))
    return eval('{}_load_saved'.format(args.framework))


def load_checkpointed_factory(args):
    return eval('{}_load_checkpoint_factory'.format(args.framework))


def load_deployed(args):
    return eval('{}_load_deployed'.format(args.framework))


def generate_model_element_factory(args):
    return eval('{}_generate_model_element'.format(args.framework))