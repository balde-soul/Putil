# coding=utf-8
import sys
import os
import datetime
import logging
import logging.handlers as logging_handlers
from colorama import Fore


HTTPHandler = logging_handlers.HTTPHandler
SMTPHandler = logging_handlers.SMTPHandler
TimedRotatingFileHandler = logging_handlers.TimedRotatingFileHandler
RotatingFileHandler = logging_handlers.RotatingFileHandler
FileHandler = logging.FileHandler
StreamHandler = logging.StreamHandler


# method
file_method = int('01', 16)
rotating_file_method = int('04', 16)
stream_method = int('02', 16)
handlers = int('0', 16)

format = logging.BASIC_FORMAT

# file name
if sys.platform == 'win32':
    user = os.environ['USERDOMAIN']
    pass
elif sys.platform == 'linux':
    user = os.environ['USER']
    pass
else:
    raise OSError("platform is not support")

# handlers default
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

# file param
file_param = {
    'file_path': './{user}-{date}-{os}.log'.format(
        user=user,
        date=datetime.date.today(),
        os=sys.platform),
    'mode': 'a',
    'encode': None,
    'delay': False
}
rotating_file_param = {
    'file_path': './{user}-{date}-{os}.log'.format(
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
    'rotating_file': logging.INFO
}


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
        self._handlers_dict = self.__set_handlers()
        for handler_key in self._handlers_dict.keys():
            if self._handlers_dict[handler_key] is not None:
                self._logger.addHandler(self._handlers_dict[handler_key])
                pass
            else:
                pass
            pass
        pass

    # todo: generate handlers and pack into dict by the keys in global log_level
    def __set_handlers(self):
        global handlers, format, file_param
        handlers_dict = {}
        if handlers & file_method != 0:
            h_file = logging.FileHandler(**file_param)
            h_file.setFormatter(format)
            h_file.setLevel(log_level['file'])
            handlers_dict['file'] = h_file
            pass
        else:
            handlers_dict['file'] = None
        if handlers & rotating_file_method != 0:
            h_file = logging_handlers.RotatingFileHandler(**rotating_file_param)
            h_file.setFormatter(format)
            h_file.setLevel(log_level['rotating_file'])
            handlers_dict['rotating_file'] = h_file
            pass
        else:
            handlers_dict['rotating_file'] = None
        if handlers & stream_method != 0:
            h_file = logging.StreamHandler()
            h_file.setFormatter(format)
            h_file.setLevel(log_level['stream'])
            handlers_dict['stream'] = h_file
            pass
        else:
            handlers_dict['stream'] = None
        return handlers_dict
        pass

    # todo: config the handles log level
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

    # todo: config the log format
    @staticmethod
    def config_format(logging_format):
        """
        :param logging_format: string
        :return:
        """
        global format
        format = logging.Formatter(logging_format)
        pass

    # todo: config which handler to use
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

    # todo: config the file handler param
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

    # todo: config the rotating file handler param
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
