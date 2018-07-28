# coding=utf-8
import sys
import os
import datetime
import logging
import logging.handlers as logging_handlers


HTTPHandler = logging_handlers.HTTPHandler
SMTPHandler = logging_handlers.SMTPHandler
TimedRotatingFileHandler = logging_handlers.TimedRotatingFileHandler
RotatingFileHandler = logging_handlers.RotatingFileHandler
FileHandler = logging.FileHandler
StreamHandler = logging.StreamHandler


file_method = int('01', 16)
rotating_file = int('04', 16)
stream_method = int('02', 16)
if sys.platform == 'win32':
    user = os.environ['USERDOMAIN']
    pass
elif sys.platform == 'linux':
    user = os.environ['USER']
    pass
else:
    raise OSError("platform is not support")

file_method_default = logging.FileHandler('./{user}-{date}-{os}.log'.format(
    user=user,
    date=datetime.date.today(),
    os=sys.platform
))
rotating_file_default = logging_handlers.RotatingFileHandler('./{user}-{date}-{os}.log'.format(
    user=user,
    date=datetime.date.today(),
    os=sys.platform
),
    maxBytes=20000000,
    encoding='utf-8'
)

log_level = logging.DEBUG
log_method = 'file'
log_file_name = './log.log'
file_handle_level = logging.NOTSET
stream_handle_level = logging.ERROR


def config_handle(logger, handle_code):

    pass


class method:
    def __init__(self):
        pass
    pass


class PutilLogConfig:
    """
    usage:
    import Putil.loger as log
    logger = log.PutilLogConfig(name).logging()
    """
    def __init__(self, name):
        self._logger = logging.getLogger(name)
        st = logging.StreamHandler()
        self._logger.setLevel(log_level)
        # self._logger.addHandler(st)
        pass

    @staticmethod
    def config_log_level(**options):
        pass

    @staticmethod
    def config_handles(**options):
        global log_method
        log_method = method
        pass

    @staticmethod
    def config_file_method(**options):
        global log_file_name
        
        pass

    @staticmethod
    def set_format(**options):
        pass

    def logging(self):
        return self._logger
