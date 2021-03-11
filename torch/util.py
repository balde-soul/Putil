# coding=utf-8

import torch

class TorchNoOpModule(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        pass

    def forward(self, x):
        return x
    pass


def set_torch_deterministic(seed):
    import torch
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    pass