# coding=utf-8
import Putil.loger as plog
import logging

plog.PutilLogConfig.config_handler(plog.file_method | plog.rotating_file_method | plog.stream_method)
plog.PutilLogConfig.config_log_level(file=logging.ERROR, stream=logging.DEBUG, rotating_file=logging.DEBUG)
logger = plog.PutilLogConfig("Test").logger()
logger.setLevel(logging.DEBUG)
logger2 = logger.getChild('sub1')
logger2.setLevel(logging.ERROR)

logger.debug("debug")
logger.info("info")
logger.warning("waning")
logger.error("error")
logger.fatal("fatal")
logger2.debug('2:fatal')
