# coding=utf-8

import tensorflow as tf
import Putil.loger as plog


root_logger = plog.PutilLogConfig("tf/optimizer").logger()
root_logger.setLevel(plog.DEBUG)
OptimizerLogger = root_logger.getChild('Optimizer')
OptimizerLogger.setLevel(plog.DEBUG)


class Optimizer:
    def __init__(self, param):
        self._param = param
        self._method = self._param['Opt']
        self._opt_param = self._param['OptParam']
        pass

    def __opt(self):
        if self._method == 'Adam':
            return self.__Adam()
        elif self._method == 'Momentum':
            return self.__Momentum()
        else:
            pass
        pass

    def __Adam(self):
        opt = tf.train.AdamOptimizer(
            self._opt_param['learning_lr'],
            self._opt_param['beta1'],
            self._opt_param['beta2'],
            self._opt_param['epsilon']
        )
        return opt
        pass

    def __AdaDelta(self):
        opt = tf.train.AdadeltaOptimizer(
            self._opt_param['learning_rate'],
            self._opt_param['rho'],
            self._opt_param['epsilon']
        )
        return opt
        pass

    def __Momentum(self):
        opt = tf.train.MomentumOptimizer(
            self._opt_param['learning_lr'],
            self._opt_param['momentum'],
            self._opt_param['use_nesterov']
        )
        return opt
        pass

    @property
    def Opt(self):
        return self.__opt()
        pass
    pass
