# coding=utf-8
import matplotlib.pyplot as plt
import Putil.function.gaussian as gaussian
import numpy as np
from matplotlib import cm

Mu = np.expand_dims(np.linspace(1, 10, 10), -1)
X = np.expand_dims(np.arange(-2, 2, 0.05), -1)
Y = np.expand_dims(np.arange(-2, 2, 0.05), -1)
X, Y = np.meshgrid(X, Y)
X = np.expand_dims(X, -1)
Y = np.expand_dims(Y, -1)
XY = np.concatenate([X, Y], axis=-1)
XY = np.transpose(np.reshape(XY, [6400, 2]))
print(X.shape)
t = gaussian.Gaussian()
t.set_Mu(np.array([[0.0], [0.0]]))
t.set_Sigma(np.array([[1.0, -0.9], [-0.9, 1.0]]))
get_p = t.func()
Z = get_p(XY)
Z = np.reshape(Z, [80, 80])
plt.contourf(np.squeeze(X), np.squeeze(Y), Z, 100, alpha = 1.0, cmap =cm.coolwarm)
plt.show()
