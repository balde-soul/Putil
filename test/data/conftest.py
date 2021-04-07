# coding=utf-8
import pytest

def pytest_addoption(parser):
    parser.addoption("--username", action="store", help="input useranme")
    parser.addoption('--cifar_root_dir', action='store', default='', help='the root dir for the cifar100')
    pass

# 解析方法
@pytest.fixture
def params(request):
    params = {}
    params['cifar_root_dir'] = request.config.getoption('--cifar_root_dir')
    return params
