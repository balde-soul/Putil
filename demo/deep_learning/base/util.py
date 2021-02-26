# coding=utf-8
import re
from enum import Enum
import shutil
import numpy as np
from colorama import Fore
import torch
import os
from Putil.base import save_fold_base as psfb
from Putil.demo.deep_learning.base import horovod

##@brief 代表着本次运行是在什么模式下
# @note 这与Stage不同，Stage在一次运行中可能会有不同的阶段Stage，
# 比如TrainEvaluate表示在RunStage.Train中的Evaluate阶段
class RunStage(Enum):
    Train=0
    Evaluate=1
    Test=2

def get_module(module_dict, target=''):
    return module_dict[target]

def find_repeatable_environ(base_name):
    temp = set([k if re.search(base_name, k) is not None else None for k in os.environ.keys()])
    temp.remove(None)
    return temp

def get_relatived_environ(base_name):
    return {property_type.replace(base_name, ''): os.environ[property_type] for property_type in find_repeatable_environ(base_name)}

def complete_environ(source_dict, target_dict, default_content):
    # 完善target_dict中缺少而source_dict中存在的类型
    [None if property_type in target_dict.keys() else target_dict.update({property_type: default_content}) \
        for property_type, name in source_dict.items()]
    pass

def empty_tensor_factory(framework, **kwargs):
    def generate_empty_tensor_factory_func():
        if framework == 'torch':
            # tensor operation
            def torch_generate_empty_tensor():
                return torch.Tensor([])
            return torch_generate_empty_tensor
        else:
            raise NotImplementedError('empty_tensor_factory in framework: {} is Not Implemented'.format(args.framework))
        pass
    return generate_empty_tensor_factory_func

def string_to_torch_tensor(_str, code='utf-16'):
    return torch.from_numpy(np.frombuffer(_str.encode(code), dtype=get_np_dtype_from_code(code)))

def get_np_dtype_from_code(code):
    return np.int16 if code == 'utf-16' else np.int8 if code == 'utf-8' else None

def get_code_from_np_dtype(np_dtype):
    return ('utf-16', np.uint16) if np_dtype == np.int16 else ('utf-8', np.uint8) if np_dtype == np.int8 else None

def torch_tensor_to_string(tensor, code='utf-16'):
    n = tensor.numpy()
    return n.astype(get_code_from_np_dtype(n.dtype)[1]).tobytes().decode(get_code_from_np_dtype(n.dtype)[0])

def make_sure_the_save_dir(name, run_stage, save_dir, weight_path, weight_epoch, debug, framework):
    hvd = horovod.horovod(framework)
    if run_stage == RunStage.Train:
        if weight_path == '' or weight_epoch is None and hvd.rank() == 0:
            bsf = psfb.BaseSaveFold(
                use_date=True if not debug else False, \
                    use_git=True if not debug else False, \
                        should_be_new=True if not debug else False, \
                            base_name='{}{}'.format(name if name is not '' else 'Unnamed', '-debug' if debug else ''))
            bsf.mkdir(save_dir)
            save_dir = bsf.FullPath
            code = 'utf-16'
            save_dir_tensor = string_to_torch_tensor(save_dir, code)
            save_dir_tensor = hvd.broadcast_object(save_dir_tensor, 0, 'save_dir')
            save_dir = torch_tensor_to_string(save_dir_tensor, code)
        elif hvd.rank() == 0 and weight_path is not None and weight_epoch is not None:
            save_dir = os.path.dirname(weight_path)
            code = 'utf-16'
            save_dir_tensor = string_to_torch_tensor(save_dir, code)
            save_dir_tensor = hvd.broadcast_object(save_dir_tensor, 0, 'save_dir')
            save_dir = torch_tensor_to_string(save_dir_tensor, code)
        elif hvd.rank() != 0:
            code = 'utf-16'
            save_dir_tensor = string_to_torch_tensor(save_dir, code)
            save_dir_tensor = hvd.broadcast_object(save_dir_tensor, 0, 'save_dir')
            save_dir = torch_tensor_to_string(save_dir_tensor, code)
        else:
            raise RuntimeError('this should not happend')
        print(Fore.GREEN + 'rank {} final get save dir: {}'.format(hvd.rank(), save_dir) + Fore.RESET)
        return save_dir
    pass

def generate_train_time_dir_name(train_time):
    return 'train_time-{}'.format(train_time)

def subdir_base_on_train_time(root_dir, train_time, prefix):
    '''
     @brief 依据根目录与对应的train_time生成子目录名称
    '''
    return os.path.join(root_dir, '{}{}'.format('' if prefix == '' else '{}-'.format(prefix), generate_train_time_dir_name(train_time)))

def train_time_matched(train_time, subdir):
    res = re.search(generate_train_time_dir_name(train_time), subdir)
    return res is not None, res

#def get_train_time_from_subdir(subdir):
#    return 

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
    #dirs.append(os.path.join(path, subdir_base_on_train_time(path, train_time)))
    # 
    _contents = os.listdir(path)
    for _content in _contents:
        matched, res = train_time_matched(train_time, _content)
        dirs.append(os.path.join(path, _content)) if matched else None
        pass
    return dirs, files

def clean_train_result(path, train_time):
    dirs, files = _get_trained_result(path, train_time)
    #sync = hvd.broadcast_object(torch.BoolTensor([True]), 0, 'sync_before_checking_remove_file')
    _remove = input(Fore.RED + 'remove the files: {} dir: {} (y/n):'.format(files, dirs))
    [os.remove(_file) for _file in files] if _remove in ['y', 'Y'] else None
    [shutil.rmtree(_dir) for _dir in dirs] if _remove in ['Y', 'y'] else None
    pass

def fix_one_env_param(param):
    if isinstance(param, bool):
        return param
    elif isinstance(param, str):
        if param in ['False', 'false']:
            return False
        elif param in ['True', 'ture']:
            return True
        elif param in ['None', 'none']:
            return None
        elif param in ['Train', 'train', 'Evaluate', 'evaluate', 'Test', 'test']:
            if param in ['Train', 'train']:
                return RunStage.Train
            elif param in ['Evaluate', 'evaluate']:
                return RunStage.Evaluate
            else:
                return RunStage.Test
            pass
        elif param in ['Torch', 'torch']:
            return 'torch'
        elif param in ['tf', 'tensorflow']:
            return 'tf'
        else:
            return param
        pass
    elif isinstance(param, None.__class__):
        return param
    else:
        raise NotImplementedError('fix param with type {} is not implemented'.format(param.__class__.__name__))
    pass

def fix_env_param(param):
    check_multi_param = param.split('.')
    if len(check_multi_param) != 1:
        temp_params = []
        for param in check_multi_param:
            temp_params.append(fix_one_env_param(param))
            pass
        return temp_params
    else:
        return fix_one_env_param(param)
    pass


def print_env_param(param, env_name):
    print(Fore.GREEN + 'param: {}:{} | type: {}'.format(env_name, param, param.__class__.__name__) + Fore.RESET)

def make_sure_the_train_time(run_stage, save_dir, framework):
    hvd = horovod.horovod(framework)
    if run_stage == RunStage.Train:
        #if hvd.rank() == 0 and args.weight_epoch is None and args.weight_epoch is None: 
        #    print(Fore.GREEN + 'get the untrained train time is 0' + Fore.RESET)
        #    args.train_time = 0
        #    train_time_tensor = torch.IntTensor([args.train_time])
        #    train_time_tensor = hvd.broadcast_object(train_time_tensor, 0, 'train_time')
        #elif hvd.rank() == 0 and (args.weight_path != '') and (args.weight_epoch is not None):
        if hvd.rank() == 0:# and (args.weight_path != '') and (args.weight_epoch is not None):
            item_list = os.listdir(save_dir)
            max_time = 0
            for _item in item_list:
                if os.path.isdir(os.path.join(save_dir, _item)):
                    name_part = _item.split('-')
                    if name_part[-2] == 'train_time':
                        max_time = max(max_time, int(name_part[-1]))
            train_time = max_time + 1
            print(Fore.GREEN + 'get the trained train time is {}'.format(train_time) + Fore.RESET)
            train_time_tensor = torch.IntTensor([train_time])
            train_time_tensor = hvd.broadcast_object(train_time_tensor, 0, 'train_time')
            train_time = train_time
        elif hvd.rank() != 0:
            print(Fore.GREEN + 'wait for the root rank share the train_time' + Fore.RESET)
            train_time_tensor = torch.IntTensor([-1])
            train_time_tensor = hvd.broadcast_object(train_time_tensor, 0, 'train_time')
            train_time = train_time_tensor.item()
        else:
            raise RuntimeError('this should not happend')
            pass
        print(Fore.GREEN + 'rank {} final get train time: {}'.format(hvd.rank(), train_time) + Fore.RESET)
        return train_time
        pass
    pass