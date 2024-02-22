# coding=utf-8

import torch
from torch.nn import Module


class ClassifyAcc(Module):
    def __init__(self, class_amount):
        Module.__init__(self)
        self._class_amount = class_amount
        pass

    def forward(self, pre, gt):
        '''
         @brief
         @note
         @param[in] x
         [0] pre [batch, class_one_hot, ...]
         [1] gt [batch, class_one_hot, ...]
        '''

        shape = pre.shape
        batch = shape[0]
        class_amount = shape[1]
        assert class_amount == self._class_amount

        pre = torch.reshape(pre, [batch, class_amount, -1])
        gt = torch.reshape(gt, [batch, class_amount, -1])

        class_weight = torch.sum(gt, -1)
        wsum = torch.sum(class_weight, -1, keepdim=True)
        eq = torch.sum(pre.eq(gt) * gt, -1)

        weighted_acc = torch.mean(torch.sum((eq * (wsum - class_weight)) / (wsum ** 2), -1), 0)
        return weighted_acc, class_weight

##In[]:
#import torch
#import numpy as np
#import random
#
#
#a = list(range(0, 10))
#
#t = torch.tensor(np.reshape(np.random.sample(100) * 10, [5, 2, 5, 2]).astype(np.int32).astype(np.float32))
#t = torch.reshape(t, [5, 2, -1])
#torch.sum(t, -1).shape
#print(torch.sum.__doc__)