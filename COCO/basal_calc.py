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

import Putil.loger as plog
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('COCO/basal_calc').logger()
root_logger.setLevel(plog.DEBUG)

coco = None


def __load_coco_dataset():

    pass


def __calc_size_distribution():
    pass


if __name__ == '__main__':
    pass
