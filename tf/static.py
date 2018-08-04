# coding = utf-8
import tensorflow as tf
import warnings
import logging


# todo: generate initialize method, coding
def initialize(method, dtype):
    """
    
    :param type: 
    :return: 
    """
    init = tf.zeros_initializer(dtype) if method == 'zero' else None
    init = tf.ones_initializer(dtype=dtype) if method == 'one' else None


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


class initializer_extract:
    def __init__(self, param):
        self._param = param
        pass

    def complement(self):
        return self

    @property
    def Param(self):
        return self._param
        pass
    pass


class regularize_extract:
    def __init__(self, param):
        """

        :param param:
        #
            {
                "regularize_weight": float,
                "regularize_way": string("l2")
            }
        #
        """
        self._param = param
        self._default = {
            'regularize_weight': 0.0,
            'regularize_way': 'l1'
        }
        self._warn_up = False
        self._check_keys()
        pass

    def _check_keys(self):
        for key in self._param.keys():
            if key in self._default.keys():
                pass
            else:
                logging.warning('key: {0} is illegal, keys must in {1}'.format(key, self._default.keys()))
                self._warn_up = True
        pass

    def complement(self, **options):
        self._param['regularize_weight'] = \
            options.pop('regularize_weight', 0.0)
        self._param['regularize_way'] = \
            options.pop('regularize_way', 'l2')
        return self

    @property
    def Param(self):
        if self._warn_up:
            return self.Default
        else:
            return self._param

    @property
    def Default(self):
        return self._default
    pass


class conv_extract:
    def __init__(self, param):
        self._param = param
        pass

    def complement(self, **options):
        self._param['kernel'] = options.pop('kernel', self._param('kernel', [3, 3]))
        self._param['stride'] = options.pop('stride', self._param.get('stride', [1, 1]))
        self._param['padding'] = options.pop('padding', self._param.get('padding', 'SAME'))
        self._param['weight_initializer'] = \
            options.pop('weight_initializer',
                        self._param.get('weight_initializer',
                                        initializer_extract({}).complement().Param))
        self._param['weight_regularize'] = \
            options.pop('weight_regularize', self._param.get('weight_regularize',
                                                             regularize_extract({}).complement().Param))

        return self

    @property
    def Param(self):
        return self._param

    def _debug_string(self):
        return ''

    @property
    def DebugInfo(self):
        return self._debug_string()

    pass
