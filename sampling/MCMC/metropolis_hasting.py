# coding=utf-8
import numpy as np
import Putil.sampling.MCMC.mcmc as mcmc


class MetropolisHasting(mcmc.MCMC):
    def __init__(self):
        mcmc.MCMC.__init__(self)
        pass

    def sample(self):
        new_x = self._random_func()
        acc = min(1, self.__get_tilde_p(new_x) / self.__get_tilde_p(self._x))
        u = np.random.random()
        if u < acc:
            self._x = new_x
            return new_x 
            pass
        else:
            return self._x
            pass
        pass
    def __get_tilde_p(self, x):
        return self._pdf_func(x) * 20
        pass
    pass


