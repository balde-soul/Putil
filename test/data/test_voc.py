# coding=utf-8

from Putil.base import logger as plog

plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(plog.FormatRecommend)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
logger = plog.PutilLogConfig('test_voc_statistic').logger()
logger.setLevel(plog.DEBUG)
TestVocStatisticLogger = logger.getChild('test_voc_statistic')
TestVocStatisticLogger.setLevel(plog.DEBUG)

import Putil.data.voc as voc
import pytest
import os

def test_voc_statistic(params):
    voc.VOC.statistic(params['voc_root'], './test/data/result/voc_statistic')
    pass

def test_voc_make_dataset(params):
    df = voc.VOC.extract_data(params['voc_root'], './test/data/result/voc_statistic')
    dataset = voc.VOC.make_dataset(df, voc.pcd.util.Stage.Train, det=True, det_sub_data=list(range(0, 5)))
    pass