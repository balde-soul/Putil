# coding=utf-8
import argparse, os, re

parser = argparse.ArgumentParser('本文件传入VOC标注格式图像路径，打印对应标记路径\n')
parser.add_argument('--image_path', dest='ImagePath', type=str, default='', action='store', help='the path of the image')
options = parser.parse_args()

_dir, tag = os.path.split(os.path.abspath(options.ImagePath))
tag = tag[0:tag.rfind('.')]
print(os.path.join(os.path.dirname(_dir), 'Annotations/{0}.xml'.format(tag)))