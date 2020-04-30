import numpy as np
import Putil.sampeling.sample as s


class Gibbs(s.Sample):
    def __init__(self):
        super(Gibbs, self).__init__()
        pass

    def sample(self, amount=1):
        pass

    def _gibbs(self):
        pass

    def _partial_sampler(self, x, dim):
        xes = []
        for t in range(10):
            xes.append(self._domain_random())
        tilde_ps = []
        for t in range(10):
            tmpx = x[:]
            tmpx[dim] = xes[t]
            tilde_ps.append(self._pdf_func(tmpx))

        norm_tilde_ps = np.asarray(tilde_ps)/sum(tilde_ps)
        u = np.random.random()
        sums = 0.0
        for t in range(10):
            sums += norm_tilde_ps[t]
            if sums>=u:
                return xes[t]
        pass

    @staticmethod
    def _domain_random():
        return np.random.random() * 3.8 - 1.9
        pass
    pass