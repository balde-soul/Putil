# coding=utf-8
import numpy as np
import Putil.data.image_data_aug.gaussian_augment as pga

aug = pga.GaussianAugment()
aug.set_config({'mu': [1.0, 0.0], 'sigma': [0.5, 0.8]})
data = np.reshape(np.random.normal(0.0, 1.0, 10000), [1, 100, 100, 1])
aug.augment(data)