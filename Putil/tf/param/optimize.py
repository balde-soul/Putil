# coding=utf-8

import Putil.loger as plog
import Putil.tf.static as param
import tensorflow as tf
from colorama import Fore, init, Style
from tensorflow.contrib import layers


root_logger = plog.PutilLogConfig('tf/optimize').logger()
root_logger.setLevel(plog.DEBUG)
# LearningRateFitParamLogger = root_logger.getChild('LearningRateFitParamLogger')
# LearningRateFitParamLogger.setLevel(plog.DEBUG)
# LearningRateFitLogger = root_logger.getChild('LearningRateFitLogger')
# LearningRateFitLogger.setLevel(plog.DEBUG)
OptimizeParamLogger = root_logger.getChild('OptimizeParamLogger')
OptimizeParamLogger.setLevel(plog.DEBUG)
OptimizeLogger = root_logger.getChild('OptimizeLogger')
OptimizeLogger.setLevel(plog.DEBUG)


# # todo: unused
# # we need to know when the learn rate reduce
# # so we need to record the epoch in the network, we add epoch update in the train method
# class LearningRateFitParam(param.ParamProbe):
#     def __init__(self, param_feed, **options):
#         """"""
#         default = options.pop('default', 'EpochSpecify')
#         epoch_specify_param_default = {
#             'method': 'EpochSpecify',
#             'learning_rate_fit_param': {
#                 'reduce_rate': [0.1, 0.1, 0.1],
#                 'reduce_epoch': [30, 60, 90]
#             }
#         }
#         default_param = {
#             "EpochSpecify": epoch_specify_param_default
#         }
#         super(LearningRateFitParam, self).__init__(default_param[default], param_feed)
#
#     @staticmethod
#     def EpochSpecify():
#         return 'EpochSpecify'
#         pass
#
#     def ShowDefault(self):
#         return super(LearningRateFitParam, self).ShowDefault(LearningRateFitParamLogger)
#     pass
#
#
# class LearningRateFit:
#     def __init__(self):
#         pass
#     pass


class OptimizeParam(param.ParamProbe):
    def __init__(self, param_feed, **options):
        """"""
        default = options.pop('default', 'Adam')
        Adam_param_default = {
            'method': 'Adam',
            'optimize_param': {
                "learning_rate": 0.001,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1e-08,
                "use_locking": False,
                "name": "Adam"
            }
        }
        AdaDelta_param_default = {
            'method': 'AdaDelta',
            'optimize_param': {
                'learning_rate': 0.001,
                'rho': 0.95,
                'epsilon': 1e-08,
                'use_locking': False,
                'name': 'AdaDelta'
            }
        }
        AdaGrad_param_default = {
            'method': 'AdaGrad',
            'optimize_param': {
                "learning_rate": 0.001,
                "initial_accumulator_value": 0.1,
                "use_locking": False,
                "name": "AdaGrad"
            }
        }
        AdaGradDA_param_default = {
            'method': 'AdaGradDA',
            'initialize_param': {
                'learning_rate': 0.001,
                'global_step': None,
                'initial_grandient_squared_accumulator_value': 0.1,
                'l1_regularization_strength': 0.0,
                'l2_regularization_strength': 0.0,
                'use_locking': False,
                'name': 'AdaGradDA'
            }
        }
        gradient_descent_param_default = {
            'method': 'GradientDescent',
            'optimize_param': {
                'learning_rate': 0.01,
                'use_locking': False,
                'name': 'GradientDescent'
            }
        }
        default_param = {
            'Adam': Adam_param_default,
            'AdaDelta': AdaDelta_param_default,
            'AdaGrad': AdaGrad_param_default,
            'AdaGradDA': AdaGradDA_param_default,
            'GradientDescent': gradient_descent_param_default
        }
        super(OptimizeParam, self).__init__(default_param[default], param_feed)
        pass

    def ShowDefault(self):
        return super(OptimizeParam, self).ShowDefault(logger=OptimizeParamLogger)
        pass
    pass


class Optimize:
    def __init__(self, param):
        self._param = param
        pass

    @property
    def Optimize(self):
        return self._optimize()

    def _optimize(self):
        OptimizeLogger.info(Fore.GREEN + 'using optimizer: {0}'.format(self._param['method']) + Style.RESET_ALL)
        OptimizeLogger.info(Fore.GREEN + 'optimizer param:\n{0}'.format(self._param['optimize_param'] + Style.RESET_ALL))
        if self._param['method'] == 'Adam':
            return self._adam_optimize()
        elif self._param['method'] == 'AdaDelta':
            return self._ada_delta_optimize()
        elif self._param['method'] == 'AdaGrad':
            return self._ada_grad_optimize()
        elif self._param['method'] == 'AdaGradDA':
            return self._ada_grad_da_optimize()
        elif self._param['method'] == 'GradientDescent':
            return self._gradient_descent_optimize()
        else:
            OptimizeLogger.error(
                Fore.RED + 'method: {0}'
            )
            raise ValueError()
            pass
        pass

    def _ada_delta_optimize(self):
        init = tf.train.AdadeltaOptimizer(
            learning_rate=self._param['optimize_param']['learning_rate'],
            rho=self._param['optimize_param']['rho'],
            epsilon=self._param
        )
        pass

    def _adam_optimize(self):
        init = tf.train.AdamOptimizer(
            learning_rate=self._param['optimize_param']['learning_rate'],
            beta1=self._param['optimize_param']['beta1'],
            beta2=self._param['optimize_param']['beta2'],
            epsilon=self._param['optimize_param']['epsilon'],
            use_locking=self._param['optimize_param']['use_locking'],
            name=self._param['optimize_param']['name']
        )
        return init
        pass

    def _ada_grad_optimize(self):
        init = tf.train.AdagradOptimizer(
            learning_rate=self._param['optimize_param']['learning_rate'],
            initial_accumulator_value=self._param['optimize_param']['initial_accumulator_value'],
            use_locking=self._param['optimize_param']['use_locking'],
            name=self._param['optimize_param']['name']
        )
        return init
        pass

    def _ada_grad_da_optimize(self):
        init = tf.train.AdagradDAOptimizer(
            learning_rate=self._param['optimize_param']['learning_rate'],
            global_step=self._param['optimize_param']['global_step'],
            initial_gradient_squared_accumulator_value=self._param['optimize_param']['initial_gradient_squared_accumulator_value'],
            l1_regularization_strength=self._param['optimize_param']['l1_regularization_strength'],
            l2_regularization_strength=self._param['optimize_param']['l2_regularization_strength'],
            use_locking=self._param['optimize_param']['use_locking'],
            name=self._param['optimize_param']['name']
        )
        return init
        pass

    def _gradient_descent_optimize(self):
        init = tf.train.GradientDescentOptimizer(
            learning_rate=self._param['optimize_param']['learning_rate'],
            use_locking=self._param['optimize_param']['use_locking'],
            name=self._param['optimize_param']['name']
        )
        return init
        pass




