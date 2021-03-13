#In[]
from Putil.base import jupyter_state
import os
jupyter_state.go_to_top(4, os.path.abspath(__file__))
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from Putil.sampling.special_sampling import gaussian_sampling

a = gaussian_sampling.GaussianSampling(2, [[0.0], [0.0]], [[0.9, 0.3], [0.3, 0.9]], range=[[-1.0, 1.0], [-1.0, 1.0]])

ss = np.linspace(-1, 1, 100)
xs = list()
for i in range(0, 20000):
    xs.append(a.sample())
    pass
xs = np.array(xs)
X, Y = np.meshgrid(ss, ss)
Z = a.pdf_func(np.stack([X, Y], axis=-1))
plt.contourf(X, Y, Z, 100, alpha=1.0, cmap=cm.coolwarm)
plt.show()
plt.contourf(X, Y, Z, 100, alpha=1.0, cmap=cm.coolwarm)
plt.scatter(xs[:, 0], xs[:, 1], c='g', marker='.', linewidths=0.0001)
plt.show()