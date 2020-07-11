# coding=utf-8
import Putil.base.logger as plog
import os
import Putil.base.arg_base as pb



def test_base_arg():
    ba = pb.ProjectArg(save_dir='./test/test_generation/test_project_base/test_base_arg', config='./test/test_generation/test_project_base/test_base_arg/config.yaml', level=plog.DEBUG, multi_gpu=[0, 1], debug=True)
    parser = ba.parser
    args = parser.parse_args()
    print(args)
    pass


if __name__ == '__main__':
    test_base_arg()
    pass
