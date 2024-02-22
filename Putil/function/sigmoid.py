# coding=utf-8
import numpy as np
import Putil.base.logger as plog
import Putil.function.function as pfunc


'''

'''
class Sigmoid(pfunc.Function):
    def __init__(self):
        pfunc.Function.__init__(self)
        self._alpha = None 
        self._ub = None
        pass

    def set_alpha(self, alpha):
        self._alpha = alpha
        pass

    def set_upper_bound(self, ub):
        self._ub = ub 
        pass

    def _build_func(self):
        def func():
            pass
        self._func = func 
        pass
    pass

##In[]:
import numpy as np
import matplotlib.pyplot as plt
import Putil.sampling.MCMC.metropolis_hasting as pmh

def rf():
    #return [np.random.choice(ss, size=1), np.random.choice(ss, size=1)]
    return [np.random.random()]
    pass

def pdff(x):
    return 1 / np.sqrt(2 * np.pi) / 0.8 * np.exp(-np.power(x[0], 2) / 0.8)
    pass

a = pmh.MetropolisHasting()
a.set_random_func(rf)
a.set_pdf_func(pdff)

data = list()
for i in range(0, 500):
    data.append(a.sample())
    pass 

plt.hist(data, bins=30)
plt.show()