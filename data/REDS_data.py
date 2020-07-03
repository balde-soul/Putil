# coding=utf-8
import cv2
import random
import os
import numpy as np
import Putil.base.logger as plog
import Putil.data.common_data as pcd
import multiprocessing

logger = plog.PutilLogConfig('REDS_data').logger()
logger.setLevel(plog.DEBUG)
REDSDataLogger = logger.getChild('REDSData')
REDSDataLogger.setLevel(plog.DEBUG)


class REDSData(pcd.CommonData):
    '''
    this dataset is a video dataset for video deblur or video super resolution
    there are five type of data in train and val.
    the "sharp" is the gt
    the "sharp_bicubic" is the type with low resolution type
    the "blur_comp" is the type which contain blur and compression
    the "blur_bicubic" is the type with low resolution and blur
    the "blur" is the type with blur

    and the test does not contain the "sharp"

    every video contain 100 frame
    '''
    def __init__(self, root):
        pcd.CommonData.__init__(self)
        self._root = root
        
        self._generate_type = None
        self._data_type = None
        self._data_name = None
        self._data_type_root = None

        # sequence
        self._sequence_len = None
        self._sequence_data_field = None
        pass

    def __check_correction(self):
        assert os.path.exists(self._root) is True, REDSDataLogger.fatal('{0} does not exist'.format(self._root))
        pass

    def __check_train_correction(self):
        pass

    def __check_val_correction(self):
        pass

    def __check_test_correction(self):
        pass

    def _restart_process(self, restart_param):
        '''
        restart_param: dict
            {'generate_type': str, sequence_len': int, 'batch_size': int, 'shuffle': bool, 'data_name': str, 'data_type': str}
            generate_type:  
            sequence_len: the number of the dim in time major
            batch_size: 
            shuffle:
            data_name: sharp_bicubic, blur_bicubic, blur_comp or blur
            data_type: train val or test
        '''
        self._data_type = restart_param['data_type']
        self._generate_type = restart_param['generate_type']
        self._data_name = restart_param['data_name']
        shuffle = restart_param['shuffle']
        self._data_type_root = os.path.join(self._root, self._data_type)
        if restart_param['generate_type'] == 'sequence':
            self._sequence_len = restart_param['sequence_len']
            self._sequence_data_field = os.listdir(os.path.join(self._data_type_root, self._data_name))
            random.shuffle(self._sequence_data_field) is restart_param['shuffle'] is True
            pass
        else:
            raise NotImplementedError('generate type {0} is not implemented'.format(restart_param['generate_type']))
        pass

    def __generate_sequence_data_field(self, data_root):
        self._sequence_data_field = os.listdir(data_root)
        pass

    def _generate_from_specified(self, index):
        if self._generate_type  == 'sequence':
            data = self._sequence_data_field[index]
            # the data
            data_path = os.path.join(os.path.join(self._data_type_root, self._data_name), data)
            frames = os.listdir(data_path)
            frames = sorted(frames)
            data = self.__read_sequence_frame(data_path, frames[0: self._sequence_len])
            gt = self.__read_sequence_frame(data_path, frames[0: self._sequence_len])
        else:
            raise NotImplementedError('generate type {0} is not implemented'.format(self._generate_type))
        return np.array([data]), 'label': np.array([gt])
        pass

    def __read_sequence_frame(self, frames_dir, sorted_frame_name_list):
        frame_collection = list()
        for frame_name in sorted_frame_name_list:
            frame_dir = os.path.join(frames_dir, frame_name)
            frame_array = cv2.imread(frame_dir)
            if len(frame_collection) != 0:
                assert frame_collection[0].shape == frame_array.shape, REDSDataLogger.fatal('ununified shape happen at frame:{0} in frames: {1}, dir:{2}'.format(frame_name, sorted_frame_name_list, frames_dir))
                pass
            frame_collection.append(frame_array)
            pass
        return np.array(frame_collection, dtype=np.uint8)
        pass

    def _data_set_field(self):
        return list(range(0, 100))
        pass

    def _inject_operation(self, inject_param):
        pass
    pass


pcd.CommonDataManager.register('REDSData', REDSData)