# coding=utf-8
import numpy as np

from Putil.function import gaussian
from Putil.sampling.MCMC import metropolis_hasting

class _RandomFunc:
    def __init__(self, dimesion, Range):
        self._dimesion = dimesion
        self._range = Range
        # Todo: do the assert
        if self._range is None:
            self._range = [[0, 1] for _dim in range(self._dimesion)]
        pass

    def __call__(self):
        return [np.random.random() * (_range[1] - _range[0]) + _range[0] for _dim, _range in zip(range(0, self._dimesion), self._range)]

def GaussianSampling(dimesion, Mu, Sigma, range=None):
    # Todo: do the assert
    g = gaussian.Gaussian()
    g.Sigma = Sigma
    g.Mu = Mu
    sampler = metropolis_hasting.MetropolisHasting()
    sampler.random_func = _RandomFunc(np.array(Mu).size, range)
    sampler.pdf_func = g
    return sampler