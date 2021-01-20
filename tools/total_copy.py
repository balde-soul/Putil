# coding=utf-8

import shutil
import os
import argparse

def _TotalCopy(source, target):
    content = os.listdir(source)
    for _content in content:
        full_dir = os.path.join(source, _content)
        if os.path.isdir(full_dir):
            _TotalCopy(full_dir, content)
        elif _content.spli('.')[-1] in ['bmp', 'png', 'jpg']:
            shutil.copyfile(os.path.join(source, _content), target)
            pass
        pass
    pass


def TotalCopy(source, target):
    project = os.path.dirname(source)
    target_dir = os.path.join(target, project)
    if os.path.exists(target_dir) is False:
        os.mkdir(target_dir)
    _TotalCopy(source, target_dir)
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, action='store', default='', \
        help='the source dir or abspath')
    parser.add_argument('--target_dir', type=str, action='store', default='', \
        help='the target dir or abspath')
    args = parser.parse_args()
    assert os.path.isdir(args.source_dir)
    source_dir = os.path.abspath(args.source_dir)
    assert os.path.isdir(args.target_dir)
    target_dir = os.path.abspath(args.target_dir)
    pass