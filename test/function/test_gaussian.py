# coding=utf-8
#In[]
import matplotlib.pyplot as plt
import Putil.function.gaussian as gaussian
import numpy as np
from matplotlib import cm

Mu = np.expand_dims(np.linspace(1, 10, 10), -1)
X = np.expand_dims(np.arange(-2, 2, 0.05), -1)
Y = np.expand_dims(np.arange(-2, 2, 0.05), -1)
X, Y = np.meshgrid(X, Y)
print(X.shape)
XY = np.stack([X, Y], axis=-1)
XY = np.reshape(XY, [-1, 2])
t = gaussian.Gaussian()
t.set_Mu([[0.0], [0.0]])
#t.set_Mu(np.array([0.0, 0.0]))
t.set_Sigma([[1.0, -0.9], [-0.9, 1.0]])
Z = t(XY)
Z = np.reshape(Z, X.shape)
plt.contourf(X, Y, Z, 100, alpha = 1.0, cmap =cm.coolwarm)
plt.show()