# coding=utf-8
import numpy as np
import Putil.function.function as function


class Gaussian(function.Function):
    '''
    the input of the function should be with shape: dim x input_amount(the amount of x)
    the return of the function is with shape: 1 x input_amount
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

    def __call__(self, x):
        de_mean = x - np.transpose(self._Mu)
        exponent = -0.5 * np.sum(np.matmul(de_mean, np.transpose(self._Sigma_inv)) * de_mean, axis=-1)
        denominator = np.ma.power(2 * np.pi, self._dim / 2.0) * np.ma.power(self._Sigma_det, 0.5) 
        ret = 1.0 / denominator * np.exp(exponent)
        return ret

    ##@brief
    # @note
    # @param[in] Sigma shape=(dim, dim), {Sigma}_ij=correlation({dim}_i, {dim}_j)
    # @return 
    def set_Sigma(self, Sigma):
        '''
         @brief
         @note
         @param[in] Sigma
         list, Sigma [[], []]
        '''
        for sigma in Sigma:
            assert len(sigma) == len(Sigma)
            pass
        self._Sigma = Sigma
        self._Sigma_inv = np.linalg.inv(Sigma)
        self._Sigma_det = np.linalg.det(Sigma)
        if self._param_confirm():
            self._config()
            pass
        else:
            self._Mu =  None
            pass
        pass

    def get_Sigma(self):
        return self._Sigma
    Sigma = property(get_Sigma, set_Sigma)

    ##@brief
    # @note 
    # @param[in] Mu shape=(dim,1)
    # @return 
    def set_Mu(self, Mu):
        for mu in Mu[1: ]:
            assert len(mu) == len(Mu[0])
        self._Mu = np.array(Mu)
        if self._param_confirm():
            self._config()
            pass
        else:
            self._Sigma = None
        pass

    def get_Mu(self):
        return self._Mu
    
    Mu = property(get_Mu, set_Mu)

    def _param_confirm(self):
        not_none = (self._Sigma is not None) and (self._Mu is not None)
        if not_none:
            return len(self._Mu) == len(self._Sigma)
        else:
            return False
        pass

    def _config(self):
        self._dim = len(self._Sigma)
        pass
    pass