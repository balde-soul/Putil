# coding=utf-8
import Putil.base.logger as plog
import os
import Putil.base.project_base as pb


def test_base_save_fold():
    root_dir = './test/test_generation/base/test_project_base'
    bsf = pb.BaseSaveFold(use_git=True, use_date=True, base_name='test_base_save_fold', should_be_new=True)
    bsf.mkdir(root_dir)
    assert os.path.split(bsf.FullPath)[0] == root_dir
    os.rmdir(bsf.FullPath)
    pass


def test_base_arg():
    ba = pb.BaseArg(save_dir='./test/test_generation/test_project_base/test_base_arg', config='./test/test_generation/test_project_base/test_base_arg/config.yaml', level=plog.DEBUG, multi_gpu=[0, 1], debug=True)
    parser = ba.Parser
    args = parser.parse_args()
    print(args)
    pass


if __name__ == '__main__':
    test_base_arg()
    pass
