import Putil.base.logger as plog
from importlib import reload

import test_reload_plog_lib

test_reload_plog_lib.a()
print('=============================')

plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
plog.PutilLogConfig.config_format(plog.FormatRecommend)
logger = plog.PutilLogConfig('b').logger()

test_reload_plog_lib.a()
print('=============================')

reload(plog)
test_reload_plog_lib.a()
print('=============================')

reload(plog)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
plog.PutilLogConfig.config_format(plog.FormatRecommend)
logger = plog.PutilLogConfig('b').logger()
reload(test_reload_plog_lib)
test_reload_plog_lib.a()
print('=============================')