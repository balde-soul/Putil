# coding=utf-8
import Putil.data.augment as paug
import numpy as np


class PepperandSaltAugment(paug.Augment):
    def __init__(self):
        paug.Augment.__init__(self)
        pass
   
    def augment(self, data):
        '''
        '''
        dc = []
        dc.append(data)
        for mu in self._config['mu']:
            for sigma in self._config['sigma']:
                noise = np.random.normal(mu, sigma, data.size)
                noise = np.reshape(noise, data.shape)
                dc.append(noise)
                pass
            pass
        ret = np.concatenate(dc, axis=0)
        return ret
        pass
    pass
