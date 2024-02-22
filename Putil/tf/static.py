# coding = utf-8
import tensorflow as tf
import warnings
import Putil.loger as plog
import copy
from colorama import Fore, Back, Style
import json
import threading

root_logger = plog.PutilLogConfig("tf/static").logger()
root_logger.setLevel(plog.DEBUG)
ParamProbeLogger = root_logger.getChild("ParamProbe")
ParamProbeLogger.setLevel(plog.DEBUG)
RegularizeExtractLogger = root_logger.getChild("RegularizeExtract")
RegularizeExtractLogger.setLevel(plog.DEBUG)


# # todo: generate initialize method, coding
# def initialize(method, dtype):
#     """
#
#     :param type:
#     :return:
#     """
#     init = tf.zeros_initializer(dtype) if method == 'zero' else None
#     init = tf.ones_initializer(dtype=dtype) if method == 'one' else None
#

# padding way describe change
class padding_convert:
    def __init__(self, padding):
        self._padding = padding
        pass

    def low(self):
        if self._padding == 'SAME':
            return 'same'
        elif self._padding == 'same':
            return 'same'
        elif self._padding == 'VALID':
            return 'valid'
        elif self._padding == 'valid':
            return 'valid'
        else:
            raise ValueError('unsupported padding way: {0}'.format(self._padding))
        pass

    @property
    def Low(self):
        return self.low()

    def high(self):
        if self._padding == 'SAME':
            return 'SAME'
        elif self._padding == 'same':
            return 'SAME'
        elif self._padding == 'VALID':
            return 'VALID'
        elif self._padding == 'valid':
            return 'VALID'
        else:
            raise ValueError('unsupported padding way: {0}'.format(self._padding))
        pass

    @property
    def High(self):
        return self.high()

    pass


def ParamUnfitInfo(param_default, param_feed):
    return 'supported param keys is: {0}\n' \
           'but input param keys are: {1}'.format(param_default.keys(), param_feed.keys())
    pass


def _DictPrint(_dict, _logger):
    info = json.dumps(_dict, indent=2)
    _logger.debug(info)
    pass


def DictPrintFormat(_dict):
    info = json.dumps(_dict, indent=2)
    return info


def CheckKeys(_param_default, _param_feed):
    """

    :param _param_default: standard param dict
    :param _param_feed: param wanted to used
    :return: raise KeyError and print unfitted information in logging.INFO
    """
    for i in _param_feed.keys():
        assert i in _param_default.keys(), \
            ParamProbeLogger.info(Fore.RED + "error key: {0}\n{1}".format(i[0], ParamUnfitInfo(
                _param_default, _param_feed)))
        if type(i[1]).__name__ == 'dict':
            pass
        pass
    pass


"""
class for fit the param request using the param we have now
param generated dose not influence the param feed in
usage:

"""


class ParamProbe:
    def __init__(self, param_default, param_feed):
        """

        :param param_default: default param construct
        :param param_feed: param now we have
        """
        self._param_default_set = threading.Lock()
        # default param
        self._param_default = param_default
        # param feed in keep no change
        self._param_feed = param_feed
        # the param we deal with
        # what we want to get
        self._param_gen = param_feed.copy()
        self._complement = ''
        self._complement_default = ''
        pass

    def SetDefault(self, param_feed):
        """
        set the _param_default
        :param param_feed:
        :return:
        """
        self._param_default_set.acquire(blocking=True)
        self._param_default = copy.deepcopy(param_feed)
        self._param_default_set.release()
        return self
        pass

    def UseDefault(self):
        """
        set _param_gen use the same as _param_default
        :return:
        """
        self._param_gen = self._param_default
        return self
        pass

    def ShowDefault(self, logger):
        """
        display param_default in Logger.debug
        :param Logger:
        :return:
        """
        _DictPrint(self._param_default, logger)
        return self
        pass

    @staticmethod
    def integrity_info():
        # debug log
        ParamProbeLogger.debug()
        pass

    def complement(self, **options):
        """
        complement the _param_gen using the special param in options
        :param options:
            key=value
            key should be in the default param keys
            value should be up to the mustard
        :return:
        """
        for i in options.items():
            self._param_gen[i[0]] = copy.deepcopy(i[1])
            self._complement += ':{0}'.format(i)
            self._complement += ']'
            pass
        return self

    # : fix the param with the default param, if the param leaked fill it
    def fix_with_default(self):
        """
        fix the param_gen using the param_default
        while the param_gen missing parts in the param_default
        just copy from param_default
        :return:
        """
        for i in self._param_default.keys():
            self._complement_default += '-'
            if type(self._param_default[i]).__name__ == 'dict':
                if self._param_gen.get(i, None) is None:
                    self._param_gen[i] = self._param_default[i]
                    self._complement_default += ':{0}'.format(i)
                else:
                    self._complement_default += ':{0}'.format(i)
                    temp = ParamProbe(self._param_default[i], self._param_gen[i])
                    self._param_gen[i] = temp.fix_with_default().ParamGen
                    self._complement_default += '{0}'.format(temp._complement_default)
                    pass
                pass
            else:
                try:
                    self._param_gen[i]
                except KeyError:
                    self._param_gen[i] = self._param_default[i]
                    self._complement_default += ':{0}'.format(i)
                    pass
                pass
            self._complement_default += ']'
            pass
        return self
        pass

    @property
    def Default(self):
        return self._param_default

    @property
    def ParamFeed(self):
        return self._param_feed

    @property
    def ParamGen(self):
        return self._param_gen

    def ParamGenWithInfo(self, logger):
        """
        info unscramble:
            display: :a]:b:e:f]:g]:h:i]]]:d]:c]], means:
            every ']' means one level out
            default:
            {'a': [1, 2, 3], 'b': {'c': 'sdsd', 'd': False, 'e': {'f': 1.0, 'g': [1, 2], 'h': {'i': 1}}}}
            param_feed:
            {'b': {'e': {'h': {}}}}
            would create this display
            :a(first level)](a element out):b(level=a):e(level=b+1):f(level=e+1)](f out)
            :g(level=f](g out)]:h(level=g):i(level=h+1)](i out)](h out)](e out):d(level=e)](d out)
            :c(level=d)](c out)](level=b)
            watch information we can make graph:
            a   =   b
            |       |
                    e   =   d   =   c
                    |
                    f   =   g   =   h
                                    |
                                    i
            and the leaky:
                a/b:e:f/b:e:g/b:e:h/b:d/b:c
        :param logger:
        :return:
        """
        logger.info(Fore.YELLOW + 'use special complement:' + Style.RESET_ALL)
        logger.info(Fore.GREEN + self._complement + Style.RESET_ALL)
        logger.info(Fore.YELLOW + 'use default' + Style.RESET_ALL)
        logger.info(Fore.GREEN + self._complement_default + Style.RESET_ALL)
        return self._param_gen

    @property
    def ComplementDefault(self):
        return self._complement_default

    @property
    def Complement(self):
        return self._complement
    pass

