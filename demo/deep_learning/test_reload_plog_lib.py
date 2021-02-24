from importlib import reload
import Putil.base.logger as plog
logger = plog.PutilLogConfig('a').logger()
logger.setLevel(plog.DEBUG)
import test_reload_plog_lib_lib as lib
# 这个非常关键
reload(lib)


def a():
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.fatal('fatal')
    lib.b()