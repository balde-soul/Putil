# coding=utf-8
from importlib import reload
import copy
import Putil.demo.deep_learning.base.base_operation as standard
import util.base_operation as project
reload(standard)
reload(project)


def checkpoint_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_checkpoint_func():
        if temp_args.framework == 'torch':
            return standard.torch_checkpoint
        else:
            raise NotImplementedError('not implemented')
        pass
    return generate_checkpoint_func


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
    temp_args = copy.deepcopy(args)
    def generate_deploy_func():
        if temp_args.framework == 'torch':
            return standard.torch_deploy
        else:
            raise NotImplementedError('not implemented')
        pass
    return generate_deploy_func


def get_models_factory(args):
    pass


def load_saved_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_load_saved_func():
        if temp_args.framework == 'torch':
            return standard.torch_load_saved
        else:
            raise NotImplementedError('not implemented')
        pass
    return generate_load_saved_func


def load_checkpointed_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_load_checkpointed_func():
        if temp_args.framework == 'torch':
            return standard.torch_load_checkpointed
        else:
            raise NotImplementedError('not implemented')
        pass
    return generate_load_checkpointed_func


def load_deployed(args):
    return eval('{}_load_deployed'.format(args.framework))

def generate_model_element_factory(args):
    return eval('{}_generate_model_element'.format(args.framework))

def empty_tensor_factory(framework, **kwargs):
    def generate_empty_tensor_factory_func():
        if framework == 'torch':
            return standard.torch_generate_empty_tensor
        else:
            raise NotImplementedError('empty_tensor_factory in framework: {} is Not Implemented'.format(args.framework))
        pass
    return generate_empty_tensor_factory_func

def combine_optimization_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_combine_optimization_factory_func():
        if temp_args.framework == 'torch':
            return standard.TorchCombineOptimization
        else:
            raise NotImplementedError('combine_optimization in framework: {} is Not Implemented'.format(args.framework))
        pass
    return generate_combine_optimization_factory_func
    pass

def is_cudable_factory(args):
    temp_args = copy.deepcopy(args)
    def generate_is_cudable_factory_func():
        if args.framework == 'torch':
            return standard.Torchis_cudable
        else:
            raise NotImplementedError('is_cudable in framework: {} is Not Implemented'.format(args.framework))
        pass
    return generate_is_cudable_factory_func
    pass