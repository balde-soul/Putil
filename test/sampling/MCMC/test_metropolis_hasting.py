#In[1]:
import matplotlib.pyplot as plt
from matplotlib import cm
import Putil.sampling.MCMC.metropolis_hasting as mh
import numpy as np


def rf():
    #return [np.random.choice(ss, size=1), np.random.choice(ss, size=1)]
    return [np.random.random() * 2 - 1, np.random.random() * 2 - 1]
    pass

def pdff(x):
    return 1 / (2 * np.pi * np.sqrt(1 - 0.85)) * np.exp(1 / (2 * (1 - 0.85)) * (- x[0]**2 - x[1]**2) * x[0] ** 2 * x[1] ** 2)
    pass
ss = np.linspace(-1, 1, 100)
a = mh.MetropolisHasting()
a.set_random_func(rf)
a.set_pdf_func(pdff)
xs = list()
for i in range(0, 20000):
    xs.append(a.sample())
    pass
xs = np.array(xs)
X, Y = np.meshgrid(ss, ss)
Z = pdff([X, Y])
plt.contourf(X, Y, Z, 100, alpha=1.0, cmap=cm.coolwarm)
plt.show()
plt.contourf(X, Y, Z, 100, alpha=1.0, cmap=cm.coolwarm)
plt.scatter(xs[:, 0], xs[:, 1], c='g', marker='.', linewidths=0.001)
plt.show()