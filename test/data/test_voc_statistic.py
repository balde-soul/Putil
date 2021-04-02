# coding=utf-8

import Putil.data.voc as voc
import pytest

def pytest_addoption(parser):
    parser.addoption('--vocroot', action='store', default='asdasd', help='')


@pytest.fixture
def vocroot(request):
    return request.config.getoption("vocroot")


def test_voc_statistic(vocroot):
    print(vocroot)
    #voc.VOC.statistic(vocroot)
    pass