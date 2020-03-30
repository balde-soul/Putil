# coding=utf-8
import sys
import os
import datetime
import logging
import logging.handlers as logging_handlers
from colorama import Fore

"""
logging 的打印机制是基于：树形结构的，上层的logger的等级以及handlers都会被下层继承
除非下层重新设置等级与handlers等属性
    loger usage:
    logger主要是联合提供了接口以及静态全局的属性，让使用该程序接口生成的logger具有相同的
    配置这在软件集成中是需要的，最上层设置输出属性，方便debug，查看等功能
        import Putil.loger as plog
        import logging
        最为简单的方法：
        plog.PutilLogConfig.config_handler(plog.file_method | plog.rotating_file_method | plog.stream_method)
        plog.PutilLogConfig.config_log_level(file=logging.ERROR, stream=logging.DEBUG, rotating_file=logging.DEBUG)
        logger = plog.PutilLogConfig("Test").logger()
        到这里，logger是继承了logging.root的level属性(logging.ERROR)
        我们可以重新设置属性
        logger.setLevel(logging.DEBUG)
        同时logger下层还可以继续生成子logger
        logger2 = logger.getChild('sub1')
        logger2.setLevel(logging.ERROR)
        同样是属性继承的，这样我们可以树形有效明了的管理log信息
"""

DEBUG = logging.DEBUG
INFO = logging.INFO
ERROR = logging.ERROR
FATAL = logging.FATAL

HTTPHandler = logging_handlers.HTTPHandler
SMTPHandler = logging_handlers.SMTPHandler
TimedRotatingFileHandler = logging_handlers.TimedRotatingFileHandler
RotatingFileHandler = logging_handlers.RotatingFileHandler
FileHandler = logging.FileHandler
StreamHandler = logging.StreamHandler


# logging.basicConfig(level=logging.INFO)

# method: the global handler for the global logger, which can be changed by PutilLogConfig.config_handlers, the default is no hanlder
no_log_handler = int('00', 16)
file_method = int('01', 16)
rotating_file_method = int('04', 16)
stream_method = int('02', 16)
handlers = int('0', 16)

Format = logging.BASIC_FORMAT
FormatRecommend = "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s"

# file name
if sys.platform == 'win32':
    user = os.environ['USERDOMAIN']
    pass
elif sys.platform == 'linux':
    user = os.environ['USER']
    pass
else:
    raise OSError("platform is not support")

# file param
file_param = {
    'filename': './{user}-{date}-{os}.log'.format(
        user=user,
        date=datetime.date.today(),
        os=sys.platform),
    'mode': 'a',
    'encoding': None,
    'delay': False
}
rotating_file_param = {
    'filename': './{user}-{date}-{os}-r.log'.format(
        user=user,
        date=datetime.date.today(),
        os=sys.platform),
    'mode': 'a',
    'maxBytes': 20000000,
    'backupCount': 0,
    'encoding': None,
    'delay': False
}

log_level = {
    'file': logging.INFO,
    'stream': logging.WARNING,
    'rotating_file': logging.DEBUG
}


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
        self._handlers_dict = self.__set_handlers()
        for handler_key in self._handlers_dict.keys():
            if self._handlers_dict[handler_key] is not None:
                self._logger.addHandler(self._handlers_dict[handler_key])
                pass
            else:
                pass
            pass
        pass

    # : generate handlers and pack into dict by the keys in global log_level
    def __set_handlers(self):
        global handlers, Format, file_param, log_level
        handlers_dict = {}
        if handlers & file_method != 0:
            h_file = logging.FileHandler(**file_param)
            h_file.setFormatter(logging.Formatter(Format))
            h_file.setLevel(log_level['file'])
            handlers_dict['file'] = h_file
            pass
        else:
            handlers_dict['file'] = None
        if handlers & rotating_file_method != 0:
            r_h_file = logging_handlers.RotatingFileHandler(**rotating_file_param)
            r_h_file.setFormatter(logging.Formatter(Format))
            r_h_file.setLevel(log_level['rotating_file'])
            handlers_dict['rotating_file'] = r_h_file
            pass
        else:
            handlers_dict['rotating_file'] = None
        if handlers & stream_method != 0:
            stream = logging.StreamHandler()
            stream.setFormatter(logging.Formatter(Format))
            stream.setLevel(log_level['stream'])
            handlers_dict['stream'] = stream
            pass
        else:
            handlers_dict['stream'] = None
        return handlers_dict
        pass

    # : config the handles log level
    @staticmethod
    def config_log_level(**options):
        """
        :param options:
        :return:
        """
        global log_level
        for key in options.keys():
            try:
                log_level[key] = options[key]
            except KeyError as e:
                logging.warning(Fore.LIGHTRED_EX + 'options key {0} is not supported in the rotating_file_param\n'
                                'keys should be: [file, stream，rotating_file]\n'
                                'ignore key: {0}'.format(key))
                pass
            pass
        pass

    # : config the log format
    @staticmethod
    def config_format(logging_format):
        """
        :param logging_format: string
        :return:
        """
        global Format
        Format = logging_format
        pass

    # : config which handler to use
    @staticmethod
    def config_handler(_handlers):
        """
        set handlers to the logger, new supporting file_method rotating_file stream_method
        :param _handlers: file_method | stream_method ...
        :return:
        """
        global handlers
        handlers = _handlers
        pass

    # : config the file handler param
    @staticmethod
    def config_file_handler(**options):
        global file_param
        for key in options.keys():
            try:
                file_param[key] = options[key]
            except KeyError as e:
                logging.warning(Fore.LIGHTRED_EX + 'options key {0} is not supported in the rotating_file_param\n'
                                'keys should be: [file_path, mode，encode, delay]\n'
                                'ignore key: {0}'.format(key))
                pass
            pass
        pass

    # : config the rotating file handler param
    @staticmethod
    def config_rotating_file_handlers(**options):
        global rotating_file_param
        for key in options.keys():
            try:
                rotating_file_param[key] = options[key]
            except KeyError as e:
                logging.warning(Fore.LIGHTRED_EX + 'options key {0} is not supported in the rotating_file_param\n'
                                'keys should be: [file_path, mode, maxBytes, backupCount, encoding, delay]'
                                'ignore key: {0}'.format(key))
                pass
            pass
        pass

    def logger(self):
        return self._logger
    pass


class LogReflect:
    def __init__(self, _level):
        """

        :param _level: string
        """
        self._level = _level
        self._reflect = {
            'Notset': logging.NOTSET,
            'Debug': logging.DEBUG,
            'Info': logging.INFO,
            'Warning': logging.WARNING,
            'Error': logging.ERROR,
            'Fatal': logging.FATAL
        }
        if self._level not in self._reflect.keys():
            raise KeyError('{0} is illegal, please set {1}'.format(
                self._level,
                self._reflect.keys().__str__()
            ))
        pass

    def level(self):
        return self._reflect[self._level]
        pass

    @property
    def Level(self):
        return self.level()
        pass
    pass


def Log(logger, level):
    if level == DEBUG:
        return logger.debuf
    elif level == INFO:
        return logger.info
    elif level == ERROR:
        return logger.error
    elif level == FATAL:
        return logger.fatal
    else:
        raise ValueError('logger level: {0}, is not supported'.format(level))
    pass


def api_function_log(logger, str):
    logger.info(Fore.GREEN + str + Fore.RESET)
    pass

def api_function_in_log(logger, str):
    logger.info(Fore.GREEN + '-->{0}'.format(str) + Fore.RESET)
    pass

def api_function_out_log(logger, str):
    logger.info(Fore.GREEN + '{0}-->'.format(str) + Fore.RESET)
    pass

def set_internal_debug_log():
    import Putil.base.logger as plog
    plog.PutilLogConfig.config_handler(plog.stream_method)
    plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
    plog.PutilLogConfig.config_format(plog.FormatRecommend)
    logger = plog.PutilLogConfig('logger').logger()
    logger.setLevel(plog.DEBUG)
    pass

'''
print with color
'''
def info(logger, message):
    logger.info(Fore.GREEN + message + Fore.RESET)
    pass

def debug(logger, message):
    logger.debug(Fore.YELLOW + message + Fore.RESET)
    pass

def error(logger, message):
    logger.error(Fore.RED + message + Fore.RESET)
    pass
'''
print with color
'''
