# coding=utf-8
from abc import ABCMeta, abstractmethod
import os
import time
import git
import argparse
import Putil.base.logger as plog
import Putil.base.arg_base as pab


ProjectBaseLogger = plog.PutilLogConfig('ProjectBase').logger()
ProjectBaseLogger.setLevel(plog.DEBUG)
ProjectArgLogger = ProjectBaseLogger.getChild('PorjectArg')
ProjectArgLogger.setLevel(plog.DEBUG)
BaseArgLogger = ProjectBaseLogger.getChild('BaseArg')
BaseArgLogger.setLevel(plog.DEBUG)
BaseSaveFoldLogger = ProjectBaseLogger.getChild('BaseSaveFoldLogger')
BaseSaveFoldLogger.setLevel(plog.DEBUG)

#parser = argparse.ArgumentParser()
#parser.add_argument('--save_dir', action='store', dest='SaveDir', default=None, help='this param specified the dir to save the result, the default is')
#
#args = parser.parse_args()
#print(args)
#args2 = parser.parse_args()
#print(args2)


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

    def show(self):
        pab.arg_log(self._parser.parse_args(), ProjectArgLogger)
        pass
    
    @property
    def parser(self):
        return self._parser
        pass
    pass

class BaseSaveFold:
    def __init__(self, **kwargs):
        '''
        use_git:
        use_date:
        base_name:
        shoudl_be_new:
        '''
        self._base_name = kwargs.get('base_name', 'base_name')
        self._use_git_info = kwargs.get('use_git', False)
        self._use_date = kwargs.get('use_date', False)
        self._should_be_new = kwargs.get('should_be_new', True)
        self._repo = git.Repo('./') if self._use_git_info is True else None
        if self._use_git_info:
            assert self._repo is not None, BaseSaveFoldLogger.fatal('this project \'{0}\'is not in git dir, please build a git repo or do not use git info'.format(os.getcwd()))
            self._repo.commit().hexsha[0:6]
        else:
            pass
        self._date = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self._name = '{0}{1}{2}'.format(self._base_name, '-{0}_{1}'.format(self._repo.active_branch.name, self._repo.commit().hexsha[0: 6]) if self._use_git_info is True else '', '-{0}'.format(self._date) if self._use_date is True else '')

        self._root_dir = None
        self._full_path = None
        self._fold_existed = None
        pass

    def mkdir(self, root_dir):
        self._root_dir = root_dir
        assert os.path.exists(self._root_dir) is True, BaseSaveFoldLogger.fatal('{0} should be existed'.format(self._root_dir))
        self._full_path = os.path.join(root_dir, self._name)
        self._fold_existed = os.path.exists(self._full_path)
        assert self._fold_existed is False, BaseSaveFoldLogger.fatal('fold: {0} should be not in the {1}, please check the path'.format(self._name, root_dir)) if self._should_be_new is True else None
        flag = os.mkdir(self._full_path) if self._should_be_new is True else os.mkdir(self._full_path) if self._fold_existed is False else None
        assert flag is None
        pass

    @property
    def Name(self):
        return self._name
        pass

    @property
    def Date(self):
        '''
        the date of the path
        '''
        return self._date

    @property
    def Repo(self):
        return self._repo
        pass

    @property
    def FullPath(self):
        '''
        the full path of the target path
        '''
        return self._full_path
        pass
    pass


if __name__ == '__main__':
    example = BaseArg('test', 'test2', save_dir='')
    pass
