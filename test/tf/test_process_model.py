# coding=utf-8
from optparse import OptionParser
parser = OptionParser(usage='usage %prog [options] arg1 arg2')
level_default = 'Debug'
parser.add_option(
    '--level',
    action='store',
    dest=Level,
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
parser.add_option(
    '--test'
)

import Putil.tf.process_model as ppm
import Putil.loger as plog
plog.PutilLogConfig.config_log_level()


if __name__ == '__main__':
    pass
