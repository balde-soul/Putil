# coding=utf-8
from sys import version_info
import numpy as np
from enum import Enum
import os
if version_info.major == 3:
    import pickle as pickle
elif version_info.major == 2:
    import cPickle as pickle

import Putil.data.common_data as pd

class Cifar(pd.CommonDataWithAug):
    class Level(Enum):
        SuperClass=0
        FineClass=1   

    def __init__(
        self,
        stage,
        root_dir,
        use_rate, 
        sub_data,
        remain_strategy,
        level,
        ):
        pd.CommonDataWithAug.__init__(self, use_rate=use_rate, sub_data=sub_data, remain_strategy=remain_strategy)
        self._root_dir = root_dir
        self._level = level
        self._stage = stage
        self._python_root_path = os.path.join(self._root_dir, 'cifar-100-python')
        self._train_file_path = os.path.join(self._python_root_path, 'train')
        self._val_file_path = os.path.join(self._python_root_path, 'val')
        self._test_file_path = os.path.join(self._python_root_path, 'test')
        self._meta_file_path = os.path.join(self._python_root_path, 'meta')
        self._data_path = self._test_file_path
        with open(self._data_path, 'rb') as fp:
            self._dict = pickle.load(fp, encoding='bytes')
        self._data_field = list(range(0, self._dict[b'data'].shape[0]))
        with open(self._meta_file_path, 'rb') as fp:
            self._meta = pickle.load(fp, encoding='bytes')
        self._class_to_name = self._meta[b'corase_label_names'] if level == Cifar100.Level.SuperClass else self._meta[b'fine_label_names']
        self._label_dict = self._dict[b'corase_labels' if level == Cifar100.Level.SuperClass else b'fine_labels']
        pass

    def _generate_from_origin_index(self, index):
        data = self._dict[b'data'][index, :]
        data = np.reshape(data, [3, 32, 32])
        label = self._label_dict[index]
        return data, label,

    def objname(self, index):
        return str(self._class_to_name[index], encoding='utf8')

##@brief
# @note
class Cifar100(Cifar):
    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    def __init__(
        self, 
        stage,
        root_dir,
        use_rate, 
        sub_data,
        remain_strategy,
        level,
        ):
        Cifar.__init__(self, stage=stage, root_dir=root_dir, use_rate=use_rate, sub_data=sub_data, remain_strategy=remain_strategy, level=level)
        pass

    def _restart_process(self, restart_param):
        '''
        process while restart the data, process in the derived class and called by restart_data
        restart_param: the argv which the derived class need, dict
        '''
        pass

    def _inject_operation(self, inject_param):
        '''
        operation while the epoch_done is False, process in the derived class and called by inject_operation
        injecct_param: the argv which the derived class need, dict
        '''
        pass
    pass