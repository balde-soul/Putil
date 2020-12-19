# coding=utf-8
from __future__ import absolute_import
import os


def deterministic_setting(seed, torch=True, numpy=True, tf=False):
    if torch:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
    if numpy:
        import numpy as np
        np.random.seed(seed)
    if tf:
        import tensorflow as tf
        raise NotImplementedError('framework tf deterministic is not implemented')
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    pass