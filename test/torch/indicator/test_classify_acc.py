# coding=utf-8

import random
import numpy as np
import torch
from Putil.torch.indicator.classify_acc import ClassifyAcc


ac = list()
for i in range(0, 100):
    pre = np.reshape(np.transpose(np.eye(4)[np.round(np.random.sample(10000) * 4.0 - 0.5).astype(np.int32)]), [1, 4, 100, 100])
    gt = np.reshape(np.transpose(np.eye(4)[np.round(np.random.sample(10000) * 4.0 - 0.5).astype(np.int32)]), [1, 4, 100, 100])
    
    pre = torch.from_numpy(pre)
    gt = torch.from_numpy(gt)
    gt
    
    ca = ClassifyAcc(4)
    ac.append(ca((pre, gt))[0].item())
    pass

print(np.mean(ac))