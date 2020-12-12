import Putil.base.logger as plog
logger = plog.PutilLogConfig('b').logger()
logger.setLevel(plog.DEBUG)


def b():
    logger.debug('debug')
    logger.info('info')
    logger.warning('warning')
    logger.error('error')
    logger.fatal('fatal')
