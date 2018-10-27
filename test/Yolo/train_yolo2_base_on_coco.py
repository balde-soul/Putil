# coding=utf-8

from pycocotools.coco import COCO
import configparser
from optparse import OptionParser

parser = OptionParser(usage='usage %prog [options] arg1 arg2')
parser.add_option(
    '--configure',
    action='store',
    dest='Configure',
    type=str,
    default='',
    help='specify the config file to the program'
)

# extract argv
(options, args) = parser.parse_args()
# extract the config
conf = configparser.ConfigParser()
conf.read(options.Configure)

LogLevel = conf.get('Program Config', 'LogLevel')

AnnDir = conf.get('COCO Config', 'AnnDir')
TrainType = conf.get('COCO Config', 'TrainType')
ValType = conf.get('COCO Config', 'ValType')
TestType = conf.get('COCO Config', 'TestType')

import Putil.loger as plog
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('train_yolo2_base_on_coco').logger()
root_logger.setLevel(LogLevel)


if __name__ == '__main__':

    pass
