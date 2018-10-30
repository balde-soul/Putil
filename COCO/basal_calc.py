# coding=utf-8

from pycocotools.coco import COCO
from optparse import OptionParser
parser = OptionParser(usage='usage %prog [options] arg1 arg2')
level_default = 'Debug'
parser.add_option(
    '--level',
    action='store',
    dest='Level',
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
coco_dir_default = ''
parser.add_option(
    '--coco_dir',
    action='store',
    dest='CocoDir',
    type=str,
    default=coco_dir_default,
    help='specify the direction of the coco data set annotations'
         'default: {0}'.format(coco_dir_default)
)
ann_file_name_default = ''
parser.add_option(
    '--ann_file_name',
    action='store',
    dest='AnnFileName',
    type=str,
    default=ann_file_name_default,
    help='specify the annotation file name list which would be calc'
         'sefault: {0}'.format(ann_file_name_default)
)
calc_default = 'all'
parser.add_option(
    '--calc',
    action='store',
    dest='Calc',
    type=str,
    default=calc_default,
    help='specify which kind of indicator to calculation'
         'default: {0}'.format(calc_default)
)
store_dir = ''
parser.add_option(
    '--store_dir',
    action='store',
    dest='StoreDir',
    type=str,
    default=store_dir,
    help='specify the path which used to store the indicator'
         'default: {0}'.format('')
)

import Putil.loger as plog
import os
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('COCO/basal_calc').logger()
root_logger.setLevel(plog.DEBUG)

coco = None


class CocoBox:
    def __init__(self, ann_file, store_dir):
        self._coco = COCO(ann_file)
        self._store_dir = store_dir
        self._image_id = self._coco.getImgIds()
        pass

    def Calc(self):
        for imgId in self._image_id:
            ann = self._coco.loadAnns(self._coco.getAnnIds(imgIds=self._coco.loadImgs(imgId)))
            
        pass


def __calc_size_distribution():

    pass


def __calc_box_indicator(ann_file, store_dir):
    cb = CocoBox(ann_file, store_dir)

    pass


class CcoColass:
    def __init__(self, ann_file, store_file):
        self._coco = COCO(ann_file)
        self._store_dir = store_dir
        pass


if __name__ == '__main__':
    ann_file = os.path.join(options.CocoDir, options.AnnFileName)
    if options.StoreDir == '':
        store_dir_box = os.path.join(options.CocoDir, 'box')
        pass
    else:
        store_dir_box = os.path.join(options.StoreDir, 'box')
        pass
    if options.Calc == 'all':
        CocoBox(ann_file, store_dir_box)
        pass
    pass
