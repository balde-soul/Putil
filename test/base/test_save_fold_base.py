# coding=utf-8
#Putil
import Putil.base.save_fold_base as psfb

import os


def test_base_save_fold():
    root_dir = './test/test_generation/base/test_save_fold'
    bsf = psfb.BaseSaveFold(use_git=True, use_date=True, base_name='test_base_save_fold', should_be_new=True)
    bsf.mkdir(root_dir)
    assert os.path.split(bsf.FullPath)[0] == root_dir
    os.rmdir(bsf.FullPath)
    pass

if __name__ == '__main__':
    test_base_save_fold()
    pass