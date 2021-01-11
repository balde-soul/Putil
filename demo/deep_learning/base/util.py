import numpy as np
import shutil
from importlib import reload
import os 
import torch
from colorama import Fore
from enum import Enum
import torch
from torch.nn import Module
import Putil.base.save_fold_base as psfb
import Putil.base.logger as plog
logger = plog.PutilLogConfig('util').logger()
logger.setLevel(plog.DEBUG)
MakeSureTheSaveDirLogger = logger.getChild('MakeSureTheSaveDir')
MakeSureTheSaveDirLogger.setLevel(plog.DEBUG)
TorchLoadCheckpointedLogger = logger.getChild('TorchLoadCheckpoited')
TorchLoadCheckpointedLogger.setLevel(plog.DEBUG)

import Putil.demo.deep_learning.base.horovod as horovod
reload(horovod)

# tensor operation
def torch_generate_empty_tensor():
    return torch.Tensor([])

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


class Stage(Enum):
    Train=0
    TrainEvaluate=1
    Evaluate=2
    Test=3


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

def get_np_dtype_from_code(code):
    return np.int16 if code == 'utf-16' else np.int8 if code == 'utf-8' else None

def get_code_from_np_dtype(np_dtype):
    return ('utf-16', np.uint16) if np_dtype == np.int16 else ('utf-8', np.uint8) if np_dtype == np.int8 else None

def string_to_torch_tensor(_str, code='utf-16'):
    return torch.from_numpy(np.frombuffer(_str.encode(code), dtype=get_np_dtype_from_code(code)))

def torch_tensor_to_string(tensor, code='utf-16'):
    n = tensor.numpy()
    return n.astype(get_code_from_np_dtype(n.dtype)[1]).tobytes().decode(get_code_from_np_dtype(n.dtype)[0])

def make_sure_the_save_dir(args):
    hvd = horovod.horovod(args)
    if train_stage(args):
        if args.weight_path == '' or args.weight_epoch is None and hvd.rank() == 0:
            bsf = psfb.BaseSaveFold(
                use_date=True if not args.debug else False, \
                    use_git=True if not args.debug else False, \
                        should_be_new=True if not args.debug else False, \
                            base_name='{}{}{}'.format(args.backbone_name, args.name, '-debug' if args.debug else ''))
            bsf.mkdir(args.save_dir)
            args.save_dir = bsf.FullPath
            code = 'utf-16'
            save_dir_tensor = string_to_torch_tensor(args.save_dir, code)
            save_dir_tensor = hvd.broadcast_object(save_dir_tensor, 0, 'save_dir')
            args.save_dir = torch_tensor_to_string(save_dir_tensor, code)
        elif hvd.rank() == 0 and args.weight_path is not None and args.weight_epoch is not None:
            args.save_dir = os.path.dirname(args.weight_path)
            code = 'utf-16'
            save_dir_tensor = string_to_torch_tensor(args.save_dir, code)
            save_dir_tensor = hvd.broadcast_object(save_dir_tensor, 0, 'save_dir')
            args.save_dir = torch_tensor_to_string(save_dir_tensor, code)
        elif hvd.rank() != 0:
            code = 'utf-16'
            save_dir_tensor = string_to_torch_tensor(args.save_dir, code)
            save_dir_tensor = hvd.broadcast_object(save_dir_tensor, 0, 'save_dir')
            args.save_dir = torch_tensor_to_string(save_dir_tensor, code)
        else:
            raise RuntimeError('this should not happend')
            pass
        print(Fore.GREEN + 'rank {} final get save dir: {}'.format(hvd.rank(), args.save_dir) + Fore.RESET)
        pass
    pass

def subdir_base_on_train_time(root_dir, train_time, prefix):
    '''
     @brief 依据根目录与对应的train_time生成子目录名称
    '''
    return os.path.join(root_dir, '{}{}'.format('' if prefix == '' else '{}-'.format(prefix), generate_train_time_dir_name(train_time)))

def generate_train_time_dir_name(train_time):
    return 'train_time-{}'.format(train_time)

def make_sure_the_train_time(args):
    hvd = horovod.horovod(args)
    if train_stage(args):
        #if hvd.rank() == 0 and args.weight_epoch is None and args.weight_epoch is None: 
        #    print(Fore.GREEN + 'get the untrained train time is 0' + Fore.RESET)
        #    args.train_time = 0
        #    train_time_tensor = torch.IntTensor([args.train_time])
        #    train_time_tensor = hvd.broadcast_object(train_time_tensor, 0, 'train_time')
        #elif hvd.rank() == 0 and (args.weight_path != '') and (args.weight_epoch is not None):
        if hvd.rank() == 0:# and (args.weight_path != '') and (args.weight_epoch is not None):
            item_list = os.listdir(args.save_dir)
            max_time = 0
            for _item in item_list:
                if os.path.isdir(os.path.join(args.save_dir, _item)):
                    name_part = _item.split('-')
                    if name_part[-2] == 'train_time':
                        max_time = max(max_time, int(name_part[-1]))
            train_time = max_time + 1
            print(Fore.GREEN + 'get the trained train time is {}'.format(train_time) + Fore.RESET)
            train_time_tensor = torch.IntTensor([train_time])
            train_time_tensor = hvd.broadcast_object(train_time_tensor, 0, 'train_time')
            args.train_time = train_time
        elif hvd.rank() != 0:
            print(Fore.GREEN + 'wait for the root rank share the train_time' + Fore.RESET)
            train_time_tensor = torch.IntTensor([-1])
            train_time_tensor = hvd.broadcast_object(train_time_tensor, 0, 'train_time')
            args.train_time = train_time_tensor.item()
        else:
            raise RuntimeError('this should not happend')
            pass
        print(Fore.GREEN + 'rank {} final get train time: {}'.format(hvd.rank(), args.train_time) + Fore.RESET)
        pass
    pass

def _tensorboard_event_and_the_train_time(file_name):
    _split_by_point = file_name.split('.')
    is_event = _split_by_point[0] == 'events' and _split_by_point[1] == 'out' and _split_by_point[2] == 'tfevents'
    train_time = int(file_name.split('-')[-1]) if is_event else None
    return is_event, train_time

def _get_trained_result(path, train_time):
    '''
     @brief 依据提供的train_time与根目录获取相关与train_time次训练的结果内容，提供给clean_train_result等进行处理
    '''
    files = []
    dirs = []
    # 
    dirs.append(os.path.join(path, subdir_base_on_train_time(path, train_time)))
    # 
    _contents = os.listdir(path)
    for _content in _contents:
        is_event, _train_time = _tensorboard_event_and_the_train_time(_content)
        files.append(os.path.join(path, _content)) if is_event and _train_time == train_time else None
        pass
    return dirs, files

def clean_train_result(path, train_time):
    dirs, files = _get_trained_result(path, train_time)
    #sync = hvd.broadcast_object(torch.BoolTensor([True]), 0, 'sync_before_checking_remove_file')
    _remove = input(Fore.RED + 'remove the files: {} dir: {} (y/n):'.format(files, dirs))
    [os.remove(_file) for _file in files] if _remove in ['y', 'Y'] else None
    [shutil.rmtree(_dir) for _dir in dirs] if _remove in ['Y', 'y'] else None
    pass

def scalar_log(logger, prefix, indicators, recorder, data_index=None, epoch_step_amount=None):
    logger.info('{0} epoch: {1}|{2} [{3}/{4}]: {5}'.format(prefix, epoch, recorder.step if epoch_step_amount is not None else '', \
        data_index if epoch_step_amount is not None else '', epoch_step_amount if epoch_step_amount is not None else '', \
            ['{}:{} '.format(k, v) for k, v in indicators.items()]))

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
        return {k: v.mean() for k, v in self._epoch_indicator.items()}

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


def all_reduce(value, name):
    if type(val).__name__ != 'Tensor':
        val = torch.tensor(val)
    avg_tensor = hvd.allreduce(val, name=name)
    return avg_tensor.item()