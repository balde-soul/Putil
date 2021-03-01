# coding=utf-8
import re
import numpy as np
import shutil
import os 
import torch
from colorama import Fore
from enum import Enum
import torch
from torch.nn import Module
from torch import optim
import Putil.base.save_fold_base as psfb
import Putil.base.logger as plog
logger = plog.PutilLogConfig('util').logger()
logger.setLevel(plog.DEBUG)
MakeSureTheSaveDirLogger = logger.getChild('MakeSureTheSaveDir')
MakeSureTheSaveDirLogger.setLevel(plog.DEBUG)
TorchLoadCheckpointedLogger = logger.getChild('TorchLoadCheckpoited')
TorchLoadCheckpointedLogger.setLevel(plog.DEBUG)
TorchIsCudableLogger = logger.getChild('IsCudable')
TorchIsCudableLogger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.horovod as horovod
from Putil.trainer import util
Stage = util.Stage

def evaluate_stage(args):
    '''
     @brief off_train is True and off_evaluate is False
    '''
    return args.train_off and not args.evaluate_off


def test_stage(args):
    '''
     @brief only run test
    '''
    return not args.test_off and args.train_off

def train_stage(args):
    '''
     @brief if off_train is False, it is in train_stage, and if the off_evaluate is False, it means the TrainEvaluate
    '''
    return not args.train_off

def iscuda(args):
    return len(args.gpus) != 0

class CombineObj:
    def __init__(self, objs):
        pass
    pass


class TorchCombineOptimization(optim.Optimizer):
    def __init__(self, optimizations):
        self._optimizations = optimizations
        optim.Optimizer.__init__(self, [param for k, optimization in self._optimizations.items() for param in optimization.param_groups for param in param['params']], {})
        pass

    def step(self):
        for k, optimization in self._optimizations.items():
            optimization.step()
            pass
        pass

    def zero_grad(self):
        for k, optimization in self._optimizations.items():
            optimization.zero_grad()
            pass
        pass

    def state_dict(self):
        return {k: optimization.state_dict() for k, optimization in self._optimizations.items()}
    
    @property
    def optimizations(self):
        return self._optimizations
    pass

def Torchis_cudable(object):
    is_cudable = isinstance(object, Module)
    TorchIsCudableLogger.info(Fore.YELLOW + 'object: {} is cudable: {}'.format(object.__module__, is_cudable) + Fore.RESET)
    return is_cudable

def ndarray_mean(ndarray_list):
    return np.mean(np.stack(ndarray_mean, axis=0), axis=0)

def tensor_mean(tensor_list):
    return torch.mean(torch.stack(tensor_list, dim=0), dim=0)

def list_mean_method(value_list):
    if isinstance(value_list[0], torch.Tensor):
        return torch.mean(torch.stack(value_list, dim=0), dim=0)
    elif isinstance(value_list[0], np.ndarray):
        return np.mean(np.stack(value_list, axis=0), axis=0)
    elif isinstance(value_list[0], (float, int)):
        return np.mean(value_list)
    elif isinstance(value_list[0], (tuple, list)):
        return np.mean(np.stack(value_list, axis=0), axis=0)
    else:
        raise NotImplementedError('mean method for {} is not defined'.format(value_list[0].__class__.__name__))
    pass

def scalar_log(logger, prefix, indicators, recorder, data_index=None, epoch_step_amount=None):
    logger.info('{0} epoch: {1}|{2} [{3}/{4}]: {5}'.format(prefix, recorder.epoch, recorder.step if epoch_step_amount is not None else '', \
        data_index if epoch_step_amount is not None else '', epoch_step_amount if epoch_step_amount is not None else '', \
            ''.join(['{}:{}||'.format(k, v) for k, v in indicators.items()])))

class nothing():
    '''this is use in with ...:'''
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    pass

class ScalarCollection:
    def __init__(self, moving_epsilon=0.1):
        self._moving_average = dict()
        self._epoch_indicator = dict()
        self._current_indicator = dict()
        self._moving_epsilon = moving_epsilon
        pass

    def batch_update(self, indicators):
        if len(self._moving_average.keys()) == 0:
            self._moving_average.clear()
            self._epoch_indicator.clear()
            self._current_indicator.clear()
            for k, v in indicators.items():
                self._moving_average[k] = 0.
                self._epoch_indicator[k] = list()
                self._current_indicator[k] = 0.
                pass
            pass
        for k, v in indicators.items():
            self._current_indicator[k] = v
            self._epoch_indicator[k].append(v)
            self._moving_average[k] = self._moving_average[k] * self._moving_epsilon + (1 - self._moving_epsilon) * v
            pass
        pass

    @property
    def moving_average(self):
        return self._moving_average

    @property
    def epoch_average(self):
        return {k: list_mean_method(v) for k, v in self._epoch_indicator.items()}

    @property
    def current_indicators(self):
        return self._current_indicator

    @staticmethod
    def generate_epoch_average_reduce_name(base_name):
        return 'epoch_mean_{}'.format(base_name)

    @staticmethod
    def generate_current_reduce_name(base_name):
        return 'current_{}'.format(base_name)

    @staticmethod
    def generate_moving_reduce_name(base_name):
        return 'moving_'.format(base_name)
    pass


def all_reduce(val, name, hvd):
    if type(val).__name__ != 'Tensor':
        val = torch.tensor(val)
    avg_tensor = hvd.allreduce(val, name=name)
    return avg_tensor

class TemplateModelDecodeCombine(Module):
    '''
     @brief the template model use in save
     @note
     @ret 关于torch使用jit.trace进行deploy时，需要model的输出是tensor，或者单元为tensor的可迭代
    '''
    def __init__(self, model, decode):
        self._model = model
        self._decode = decode
        pass
    pass

def torch_generate_model_element(epoch):
    '''
     @brief use the element from the result of get_all_model to generate the target model name
     @ret dict represent the information useful in evaluate and test
    '''
    return {'checkpoint': torch_generate_checkpoint_name(epoch), 'deploy': torch_generate_deploy_name(epoch), \
        'save': torch_generate_save_name(epoch)}

def torch_generate_model_epoch(file_name):
    return file_name.split('.')[0].split('-')[0]

def torch_target_model_filter(file_name):
    '''
     @brief check the file_name is the file or not
     @ret bool
    '''
    if file_name.split('.')[-1] == 'pt':
        return True
    else:
        return False


def torch_get_all_model(target_path):
    '''
     @brief 
    '''
    elements = list()
    epochs = list()
    files = os.listdir(target_path)
    for _file in files:
        if torch_target_model_filter(_file) is True:
            epochs.append(int(torch_generate_model_epoch(_file)))
        else:
            continue
    epochs = sorted(epochs)
    for epoch in epochs[::-1]:
        me = torch_generate_model_element(epoch)
        elements.append(me)
    return {'epochs': epochs, 'elements': elements}


def torch_generate_deploy_name(epoch):
    return '{}-traced_model-jit.pt'.format(epoch)


def torch_generate_checkpoint_name(epoch):
    return '{}-checkpoint.pkl'.format(epoch)


def torch_generate_save_name(epoch):
    return '{}-save.pth'.format(epoch)


def torch_deploy(template_model_decode_combine, input_example, epoch, full_path, *modules, **kwargs):
    # :use JIT to deploy a model
    '''
     @brief deploy model using in other language, such as c++, lua
     @note use JIT to deploy the model, we should combine the model and decode in to one Module
     @param[in] model the model
     @param[in] decode the decode
     @param[in] template_model_decode_combine the Module which inherit from TemplateModelDecodeCombine
     @param[in] input_example the input_example
     @param[in] epoch
     @param[in] full_path
    '''
    target_path = os.path.join(full_path, torch_generate_deploy_name(epoch))
    logger.info(Fore.BLUE + 'deploy to {}'.format(target_path) + Fore.RESET)
    traced_script_module = torch.jit.trace(template_model_decode_combine(*modules, **kwargs), input_example)
    traced_script_module.save(target_path)
    pass


def torch_save(template_model_decode_combine, epoch, full_path, *modules, **kwargs):
    '''
     @brief deploy model using in evaluate stage
     @note use torch.save to save the model, we should combine the model and decode in to one Module
     @param[in] model the model
     @param[in] decode the decode
     @param[in] template_model_decode_combine the Module which inherit from TemplateModelDecodeCombine
     @param[in] epoch
     @param[in] full_path
    '''
    target_path = os.path.join(full_path, torch_generate_save_name(epoch))
    logger.info(Fore.BLUE + 'save to {}'.format(target_path) + Fore.RESET)
    torch.save(template_model_decode_combine(*modules, **kwargs), target_path)
    pass


def torch_checkpoint(epoch, full_path, *kargs, **kwargs):
    '''
     @brief deploy model using in continue training
     @note use torch.save to save the state_dict
     @param[in] model the model
     @param[in] decode the decode
     @param[in] template_model_decode_combine the Module which inherit from TemplateModelDecodeCombine
     @param[in] epoch
     @param[in] full_path
    '''
    target_path = os.path.join(full_path, torch_generate_checkpoint_name(epoch))
    logger.info(Fore.BLUE + 'checkpoint to {}'.format(target_path) + Fore.RESET)
    state_dict = {key: value.state_dict() for key, value in kwargs.items()}
    torch.save(state_dict, target_path)
    pass


def torch_load_saved(epoch, full_path, *args, **kwargs):
    target_saved = os.path.join(full_path, torch_generate_save_name(epoch))
    logger.info(Fore.BLUE + 'load saved from {}'.format(target_saved) + Fore.RESET)
    model = torch.load(target_saved, **kwargs)
    return model


def torch_load_checkpointed(epoch, full_path, target_modules, **kwargs):
    '''
     @param[in] target_modules the dict with {name: module_obj}, aligning with torch_checkpoint
    '''
    target_checkpointed = os.path.join(full_path, torch_generate_checkpoint_name(epoch))
    logger.info(Fore.BLUE + 'load checkpointed from {}'.format(target_checkpointed) + Fore.RESET)
    state_dict = torch.load(target_checkpointed, map_location=kwargs.get('map_location', None))
    for module_name, module in target_modules.items():
        if module_name not in state_dict.keys():
            TorchLoadCheckpointedLogger.warning(Fore.RED + '{} is not in the state_dict'.format(module_name) + Fore.RESET)
        else:
            TorchLoadCheckpointedLogger.info(Fore.GREEN + 'load {}'.format(module_name) + Fore.RESET)
            eval('module.load_state_dict(state_dict[\'{}\'])'.format(module_name))
        pass
    pass


def torch_load_deploy(epoch, full_path):
    raise NotImplementedError('this should not called in python')


#class Stage(Enum):
#    Train=0
#    TrainEvaluate=1
#    Evaluate=2
#    Test=3
