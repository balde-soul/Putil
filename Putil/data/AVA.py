# coding=utf-8
import os, sys
import Putil.base.logger as plog

logger = plog.PutilLogConfig('{0}'.format(os.path.split(__file__)[-1].split('.')[0])).logger()
logger.setLevel(plog.DEBUG)
AVALogger = logger.getChild('AVADataset')
AVALogger.setLevel(plog.DEBUG)
import Putil.data.common_data as pcd

class AVADataset(pcd.CommonDataForTrainEvalTest):
    def __init__(self):
        pass
    pass