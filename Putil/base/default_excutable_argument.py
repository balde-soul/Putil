# coding=utf-8
import argparse
import os
import base.base_information as bi

class Argument:
    def __init__(self, log_level=True, log_file=True):
        self._log_file = log_file
        self._log_level = log_level
        self._parser = argparse.ArgumentParser()
        self._parser.add_argument(
            '--level',
            action='store',
            dest='Level',
            type=str,
            default='Debug',
            help='specify the log level, default is : {0}'.format('Debug')
        ) if self._log_level is True else None

        self._user_home = bi.UserHome
        self._excutable_file_name = bi.ExcutableName
        self._date = bi.Date
        if self._log_file:
            self._log_dir = os.path.join(self._user_home, 'log')
            os.mkdir(self._log_dir) if os.path.exists(self._log_dir) is False else None
            self._excutable_log_dir = os.path.join(self._log_dir, self._excutable_file_name)
            os.mkdir(self._excutable_log_dir) if os.path.exists(self._excutable_log_dir) is False else None
            self._log_file_name = '{1}{2}{3}{4}{5}{6}.{0}'.format('log', self._date.year, self._date.month,
                                                                               self._date.day,
                                                                               self._date.hour, self._date.minute,
                                                                               self._date.second)
            self._log_file_path = os.path.join(self._excutable_log_dir, self._log_file_name)
            self._parser.add_argument(
                '--log_dir',
                action='store',
                dest='LogDir',
                type=str,
                default=self._log_file_path,
                help='specify the lof file , default is : {0}'.format(self._log_file_path)
            )
        pass

    @property
    def parser(self):
        return self._parser
        pass

    @property
    def log_file_path(self):
        if self._log_file:
            return self._log_file_path
        raise AttributeError('do not support log file ')


    @property
    def log_file_name(self):
        if self._log_file:
            return self._log_file_name
        raise AttributeError('do not support log file ')


    @property
    def log_file_dir(self):
        if self._log_file:
            return self._excutable_log_dir
        raise AttributeError('do not support log file ')

    @property
    def bi(self):
        return bi
