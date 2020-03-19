# coding=utf-8
import numpy as np
import Putil.function.function as function


class Gaussian(function.Function):
    '''
    the input of the function should be with shape: input_amount x dim
    the return of the function is with shape: input_amount x 1
    '''
    def __init__(self):
        function.Function.__init__(self)
        self._func = None
        self._Sigma = None
        self._Sigma_inv = None
        self._Sigma_det = None
        self._Mu = None
        self._dim = None
        pass

    def set_Sigma(self, Sigma):
        '''
        params:
            Sigma: should with shape: dim x dim: var_ij
        '''
        assert len(Sigma.shape) == 2
        assert Sigma.shape[0] == Sigma.shape[1]
        self._Sigma = Sigma
        self._Sigma_inv = np.linalg.inv(Sigma)
        self._Sigma_det = np.linalg.det(Sigma)
        if self._param_confirm():
            self._build_func()
            pass
        else:
            self._Mu =  None
            pass
        pass

    def set_Mu(self, Mu):
        '''
        params:
            Mu: should with shape: dim x 1: mu_i
        '''
        assert len(Mu.shape) == 2
        assert Mu.shape[1] == 1
        self._Mu = Mu
        if self._param_confirm():
            self._build_func()
            pass
        else:
            self._Sigma = None
        pass

    def _param_confirm(self):
        not_none = (self._Sigma is not None) and (self._Mu is not None)
        if not_none:
            return self._Mu.shape[0] == self._Sigma.shape[0]
            pass
        else:
            return False
            pass
        pass

    def _build_func(self):
        self._dim = self._Sigma.shape[0]
        def func(x):
            return 1.0 / (np.ma.power(2 * np.pi, self._dim / 2.0) * np.ma.power(self._Sigma_det, 0.5)) * \
                np.exp(-0.5 * np.sum(np.matmul(np.transpose((x - self._Mu)), self._Sigma_inv) * np.transpose((x - self._Mu)), axis=-1, keepdims=True))
            pass
        self._func = func
        pass
    pass
