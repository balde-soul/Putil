# coiding=utf-8
# Putil
import git
import time
import os


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
            assert self._repo is not None, print('this project \'{0}\'is not in git dir, please build a git repo or do not use git info'.format(os.getcwd()))
            self._repo.commit().hexsha[0:6]
        else:
            pass
        self._date = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
        self._name = '{0}{2}{1}'.format(self._base_name, '-{0}_{1}'.format(self._repo.active_branch.name, self._repo.commit().hexsha[0: 6]) if self._use_git_info is True else '', '-{0}'.format(self._date) if self._use_date is True else '')

        self._root_dir = None
        self._full_path = None
        self._fold_existed = None
        pass

    def mkdir(self, root_dir):
        self._root_dir = root_dir
        assert os.path.exists(self._root_dir) is True, print('{0} should be existed'.format(self._root_dir))
        self._full_path = os.path.join(root_dir, self._name)
        self._fold_existed = os.path.exists(self._full_path)
        assert self._fold_existed is False, print('fold: {0} should be not in the {1}, please check the path'.format(self._name, root_dir)) if self._should_be_new is True else None
        flag = os.mkdir(self._full_path) if self._should_be_new is True else os.mkdir(self._full_path) if self._fold_existed is False else None
        assert flag is None
        pass

    def try_mkdir(self, root_dir):
        self._root_dir = root_dir
        assert os.path.exists(self._root_dir) is True, print('{0} should be existed'.format(self._root_dir))
        self._full_path = os.path.join(root_dir, self._name)
        self._fold_existed = os.path.exists(self._full_path)
        if self._fold_existed is False:
            print('fold: {0} should be not in the {1}, please check the path'.format(self._name, root_dir)) if self._should_be_new is True else None
        else:
            flag = os.mkdir(self._full_path) if self._should_be_new is True else os.mkdir(self._full_path) if self._fold_existed is False else None
            assert flag is None
            pass
        pass

    @property
    def Name(self):
        return self._name

    @property
    def Date(self):
        '''
        the date of the path
        '''
        return self._date

    @property
    def Repo(self):
        return self._repo

    @property
    def FullPath(self):
        '''
        the full path of the target path
        '''
        return self._full_path
    pass
