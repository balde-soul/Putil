# coding=utf-8
from __future__ import absolute_import


def deterministic_setting(seed, torch=True, numpy=True, tf=False):
    if torch:
        import torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    if numpy:
        import numpy as np
        np.random.seed(args.seed)
    if tf:
        import tensorflow as tf
        raise NotImplementedError('framework tf deterministic is not implemented')
    random.seed(args.seed)
    pass