# coding=utf-8

import Putil.loger as plog
import Putil.tf.static as param
import tensorflow as tf
from colorama import Fore, init, Style
from tensorflow.contrib import layers


root_logger = plog.PutilLogConfig('tf/batch_normal').logger()
root_logger.setLevel(plog.DEBUG)
BatchNormalLogger = root_logger.getChild('BatchNormal')
BatchNormalLogger.setLevel(plog.DEBUG)
BatchNormalParamLogger = root_logger.getChild('BatchNormalParam')
BatchNormalParamLogger.setLevel(plog.DEBUG)


class BatchNormalParam(param.ParamProbe):
    def __init__(self, param_feed, **options):
        self.default = options.pop('default', 'normal')
        normal_param_default = {
            'method': 'normal',
            'type': 0.32,
            'batch_normal_param': {

            }
        }
        default_param = {
            'normal': normal_param_default
        }
        param.ParamProbe.__init__(self, default_param[self.default], param_feed)
        pass

    @staticmethod
    def NORMAL():
        return 'normal'
    pass


class BatchNormal:
    def __init__(self, param, input):
        self._param = param
        self._input = input
        pass

    def LayerOutput(self):
        if self._param['method'] == BatchNormalParam.NORMAL():
            return self.__normal()
        else:
            BatchNormalLogger.error(Fore.RED +
                                    'method: {0} is not supported'
                                    + Fore.RESET
                                    )
            raise ValueError()
        pass

    def __normal(self):
        layers.batch_norm(
            self._input,
            decay=self._param,
        )
        pass

    pass

