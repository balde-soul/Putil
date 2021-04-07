import pytest

# 增加传入参数配置
def pytest_addoption(parser):
    parser.addoption("--username", action="store", help="input useranme")
    parser.addoption("--password", action="store", help="input password")

# 解析方法
@pytest.fixture
def params(request):
    params = {}
    params['username'] = request.config.getoption('--username')
    params['password'] = request.config.getoption('--password')
    return params
