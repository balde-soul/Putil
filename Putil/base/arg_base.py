# coding=utf-8
# this file is called before logger config
# Putil
import Putil.base.dict_base as pdb
from abc import ABCMeta, abstractmethod
import argparse

class ProjectArg(metaclass=ABCMeta):
    def __init__(self, parser=None, *args, **kwargs):
        self._parser = argparse.ArgumentParser() if parser is None else parser
        self._save_dir = kwargs.get('save_dir', None)
        self._level = kwargs.get('log_level', None)
        self._debug = kwargs.get('debug_mode', None)
        self._config = kwargs.get('config', None)
        self._parser.add_argument('--save_dir', action='store', dest='save_dir', default=self._save_dir, help='this param specified the dir to save the result, the default is {0}'.format(self._save_dir)) if self._save_dir is not None else None
        self._parser.add_argument('--log_level', action='store', dest='log_level', default=self._level, help='this param specified the log level, the default is {0}'.format(self._level)) if self._level is not None else None
        self._parser.add_argument('--debug_mode', action='store_true', dest='debug_mode', default=self._debug, help='this param set the program mode if the program contain a debug method, the default is {0}'.format(self._debug)) if self._debug is True else None
        self._parser.add_argument('--config', action='store', dest='config', default=self._config, help='this param set the config file path for the program if needed, the default is {0}'.format(self._config)) if self._config is not None else None
        pass
    
    @property
    def parser(self):
        return self._parser
    pass


def args_pack(args):
    '''
    this function pack the args into a string with format: key1-value1_key2-value2
    '''
    collection = args.__dict__
    return pdb.dict_back(collection)


def args_log(args, logger):
    collection = args.__dict__
    pdb.dict_log(collection, logger)
    pass

from argparse import Action

##@brief
# @note:
#    argparse action to split an argument into KEY=VALUE form
#    on the first = and append to a dictionary. List options can
#    be passed as comma separated values, i.e 'KEY=V1,V2,V3', or with explicit
#    brackets, i.e. 'KEY=[V1,V2,V3]'. It also support nested brackets to build
#    list/tuple values. e.g. 'KEY=[(V1,V2),(V3,V4)]'
# @time 2023-03-17
# @author cjh
class DictAction(Action):
    @staticmethod
    def _parse_int_float_bool(val):
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        if val.lower() in ['true', 'false']:
            return True if val.lower() == 'true' else False
        if val == 'None':
            return None
        return val

    ##@brief
    # @note
    #    Parse iterable values in the string.
    #    All elements inside '()' or '[]' are treated as iterable values.
    #    Args:
    #        val (str): Value string.
    #    Returns:
    #        list | tuple: The expanded list or tuple from the string.
    #    Examples:
    #        >>> DictAction._parse_iterable('1,2,3')
    #        [1, 2, 3]
    #        >>> DictAction._parse_iterable('[a, b, c]')
    #        ['a', 'b', 'c']
    #        >>> DictAction._parse_iterable('[(1, 2, 3), [a, b], c]')
    #        [(1, 2, 3), ['a', 'b'], 'c']
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-03-17
    # @author cjh
    @staticmethod
    def _parse_iterable(val):
        def find_next_comma(string):
            """Find the position of next comma in the string.

            If no ',' is found in the string, return the string length. All
            chars inside '()' and '[]' are treated as one element and thus ','
            inside these brackets are ignored.
            """
            assert (string.count('(') == string.count(')')) and (
                    string.count('[') == string.count(']')), \
                f'Imbalanced brackets exist in {string}'
            end = len(string)
            for idx, char in enumerate(string):
                pre = string[:idx]
                # The string before this ',' is balanced
                if ((char == ',') and (pre.count('(') == pre.count(')'))
                        and (pre.count('[') == pre.count(']'))):
                    end = idx
                    break
            return end

        # Strip ' and " characters and replace whitespace.
        val = val.strip('\'\"').replace(' ', '')
        is_tuple = False
        if val.startswith('(') and val.endswith(')'):
            is_tuple = True
            val = val[1:-1]
        elif val.startswith('[') and val.endswith(']'):
            val = val[1:-1]
        elif ',' not in val:
            # val is a single value
            return DictAction._parse_int_float_bool(val)

        values = []
        while len(val) > 0:
            comma_idx = find_next_comma(val)
            element = DictAction._parse_iterable(val[:comma_idx])
            values.append(element)
            val = val[comma_idx + 1:]
        if is_tuple:
            values = tuple(values)
        return values

    def __call__(self, parser, namespace, values, option_string=None):
        options = {}
        for kv in values:
            key, val = kv.split('=', maxsplit=1)
            options[key] = self._parse_iterable(val)
        setattr(namespace, self.dest, options)