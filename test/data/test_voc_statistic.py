# coding=utf-8

import Putil.data.voc as voc
import pytest

def test_voc_statistic(params):
    voc.VOC.statistic(params['voc_root'])
    pass