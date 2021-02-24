# coding=utf-8
import numpy as np
from colorama import Fore
import torch
import os
from Putil.base import save_fold_base as psfb
from Putil.demo.deep_learning.base import horovod

def subdir_base_on_train_time(root_dir, train_time, prefix):
    '''
     @brief 依据根目录与对应的train_time生成子目录名称
    '''
    return os.path.join(root_dir, '{}{}'.format('' if prefix == '' else '{}-'.format(prefix), generate_train_time_dir_name(train_time)))

def generate_train_time_dir_name(train_time):
    return 'train_time-{}'.format(train_time)


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

def torch_tensor_to_string(tensor, code='utf-16'):
    n = tensor.numpy()
    return n.astype(get_code_from_np_dtype(n.dtype)[1]).tobytes().decode(get_code_from_np_dtype(n.dtype)[0])

def make_sure_the_save_dir(name, stage, save_dir, weight_path, weight_epoch, debug, framework):
    hvd = horovod.horovod(framework)
    if stage == 'Train':
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
        pass
    pass