# coding=utf-8
import tensorflow as tf
import json
import os


class GenerateSessionConfig:
    def __init__(self):
        self._config = None
        self._gpu_config = None
        pass

    def set_gpu_list(self, gpu_list):
        self._gpu_config = tf.GPUOptions()
        pass

    def from_file(self, file_path):
        pass

    def from_dict(self, file_path):
        pass

    def default(self):
        pass

    @property
    def config(self):
        pass
    pass
