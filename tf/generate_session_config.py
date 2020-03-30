# coding=utf-8
import tensorflow as tf
import json
import os


class GenerateSessionConfig:
    '''
    the config is defined in the core/protobuf/config.proto
    '''
    def __init__(self):
        self._config = None
        self._gpu_config = None
        pass

    def set_gpu_list(self, gpu_list):
        self._gpu_config = tf.GPUOptions()
        pass

    def from_file(self, file_path):
        pass

    def from_dict(self, config):
        if 'CUDA_VISIBLE_DEVICES' in config.keys():
            os.environ['CUDA_VISIBLE_DEVICES'] = config['CUDA_VISIBLE_DEVICES']
        GPUConfig = tf.GPUOptions()
        GPUConfig.allow_growth = config['GPU_ALLOW_GROWTH'] if 'GPU_ALLOW_GROWTH' in config.keys() else GPUConfig.allow_growth
        self._config = tf.ConfigProto(gpu_options=GPUConfig)
        pass

    def default(self):
        pass

    @property
    def config(self):
        return self._config
        pass
    pass
