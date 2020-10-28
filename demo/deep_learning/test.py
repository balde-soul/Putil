'''
针对目标路径下的deploy模型，基于测试集计算得到结果
'''
import os


if __name__ == '__main__':
    import argparse
    options = argparse.ArgumentParser()
    options.add_argument('--target_path', type=str, default='', action='store', \
        help='the path where model saved')
    options.add_argument('--target_dataset', type=str, default='', action='store', \
        help='run test base on this dataset')
    options.add_argument('--specified_model', nargs='+', default=[], \
        help='only run with the specified model, if not set, run all model in the target path')
    args = options.parse_args()
    import Putil.base.logger as plog
    test_logger = plog.PutilLogConfig('test').logger()
    test_logger.setLevel(plog.DEBUG)
    MainLogger = test_logger.getChild('Main')
    MainLogger.setLevel(plog.DEBUG)

    assert args.target_path != '', MainLogger.fatal('')

    from util.util import get_all_model
    elements = get_all_model(args.target_path)