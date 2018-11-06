# coding=utf-8
from optparse import OptionParser
parser = OptionParser(usage="usage %prog [options] arg1 arg2")
level_default = 'Debug'
parser.add_option(
    '--Level',
    action='store',
    type=str,
    help='set the log level'
         'default: {0}'.format(level_default)
)
test_scatter_hist_default = False
parser.add_option(
    '--test_scatter_hist',
    action='store_true',
    default=test_scatter_hist_default,
    help='if you want to test scatter_hist, set this flag'
         'default: {0}'.format(test_scatter_hist_default)
)

(options, args) = parser.parse_args()
import Putil.base.logger as plog
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('test/calc/test_estimate').logger()
root_logger.setLevel(plog.DEBUG)

ScatterHistLogger = root_logger.getChild("ScatterHist")
ScatterHistLogger.setLevel(plog.DEBUG)

def ScatterHist():
    pass

if __name__ == '__main__':

    pass