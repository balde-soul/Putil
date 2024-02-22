# coding=utf-8
import shutil
import os
import argparse

options = argparse.ArgumentParser()
options.add_argument('--source_path', type=str, action='store', default='', help='the source path')
options.add_argument('--target_path', type=str, action='store', default='', help='the target path')
options.add_argument('--avoid_fold_name', type=str, nargs='+', default=[], help='only copy the fold')
args = options.parse_args()
args.source_path = '/home/caojihua/Download/note'
args.target_path = '/home/caojihua/Download/note-c'
args.avoid_fold_name = ['.git']


def wrong_markdown_format(file_name):
    split_with_dot = file_name.split('.')
    wrong_format = False
    now_format = ''
    if len(split_with_dot) != 1:
        if split_with_dot[-1] in ['txt']:
            wrong_format = True
            now_format = '.{}'.format(split_with_dot[-1])
        elif split_with_dot[-1] == 'md':
            wrong_format = False
        pass
    else:
        wrong_format = True
        now_format = ''
    return wrong_format, '' 

def not_avoid(fold):
    return fold not in args.avoid_fold_name

def deal_dir(source_path, target_dir):
    contents = os.listdir(source_path)
    for content in contents:
        spath = os.path.join(source_path, content)
        tpath = os.path.join(target_dir, content)
        if os.path.isdir(spath):
            if not_avoid(content):
                os.mkdir(tpath)
                deal_dir(spath, tpath)
                pass
            else:
                shutil.copytree(spath, tpath)
                pass
            pass
        else:
            wrong_format, now_format = wrong_markdown_format(content)
            if not wrong_format:
                shutil.copyfile(spath, tpath)
                pass
            else:
                content = content.replace(now_format, '')
                shutil.copyfile(spath, os.path.join(target_dir, '{}.md'.format(content)))
                pass
            pass
        pass
    pass

if __name__ == '__main__':
    assert args.source_path != ''
    assert args.target_path != ''
    assert os.path.exists(args.source_path) and os.path.isdir(args.source_path)
    assert not os.path.exists(args.target_path)
    os.mkdir(args.target_path)
    deal_dir(args.source_path, args.target_path)
    pass