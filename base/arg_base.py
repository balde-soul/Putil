# coding=utf-8
# this file is called before logger config
# Putil
import Putil.base.logger as plog
import Putil.base.dict_base as pdb

from abc import ABCMeta, abstractmethod
import argparse

class ProjectArg(metaclass=ABCMeta):
    def __init__(self, parser=None, *args, **kwargs):
        self._parser = argparse.ArgumentParser() if parser is None else parser
        self._save_dir = kwargs.get('save_dir', None)
        self._level = kwargs.get('log_level', None)
        self._debug = kwargs.get('debug_mode', None)
        self._config = kwargs.get('config', None)
        self._parser.add_argument('--save_dir', action='store', dest='SaveDir', default=self._save_dir, help='this param specified the dir to save the result, the default is {0}'.format(self._save_dir)) if self._save_dir is not None else None
        self._parser.add_argument('--log_level', action='store', dest='Level', default=self._level, help='this param specified the log level, the default is {0}'.format(self._level)) if self._level is not None else None
        self._parser.add_argument('--debug_mode', action='store_true', dest='DebugMode', default=self._debug, help='this param set the program mode if the program contain a debug method, the default is {0}'.format(self._debug)) if self._debug is True else None
        self._parser.add_argument('--config', action='store', dest='Config', default=self._config, help='this param set the config file path for the program if needed, the default is {0}'.format(self._config)) if self._config is not None else None
        pass
    
    @property
    def parser(self):
        return self._parser
        pass
    pass


def args_pack(args):
    '''
    this function pack the args into a string with format: key1-value1_key2-value2
    '''
    collection = args.__dict__
    return pdb.dict_back(collection)
    pass


def args_log(args, logger):
    collection = args.__dict__
    pdb.dict_log(collection, logger)
    pass