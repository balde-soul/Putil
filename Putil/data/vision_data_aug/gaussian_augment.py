# coding=utf-8
import numpy as np
import time

# Putil
import data.augment as paug
import base.logger as plog


GaussianAugmentLogger = plog.PutilLogConfig('gaussian_augment').logger()
GaussianAugmentLogger.setLevel(plog.DEBUG)


class GaussianAugment(paug.Augment):
    '''
    config:
        mu: []
        sigma: []
    '''
    def __init__(self):
        paug.Augment.__init__(self)
        pass
   
    def augment(self, data, label=None):
        '''
        data: [batch, height, width, channel]
        label: [batch, *]
        '''
        dc = []
        dc.append(data)
        for mu in self._config['mu']:
            for sigma in self._config['sigma']:
                #np.random.seed((time.time()))
                noise = np.random.normal(mu, sigma, data.size)
                noise = np.reshape(noise, data.shape)
                dc.append(noise)
                pass
            pass
        ret = np.concatenate(dc, axis=0)
        return ret
        pass
    pass

#In[]:
#import numpy as np
#
#a = np.zeros(shape=[1, 1, 10, 1, 10, 1])
#np.squeeze(a).shape