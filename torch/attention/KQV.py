# coding=utf-8

import torch

from Putil.torch.functional import correlated_weight
from Putil.torch.util import TorchNoOpModule
from Putil.torch import _functional
from Putil.torch import functional


class KQV(torch.nn.Module):
    def __init__(self, dims, key_func, query_func, value_func):
        torch.nn.Module.__init__(self)
        self._dims = dims
        self._key_func = key_func
        self._query_func = query_func
        self._value_func = value_func
        self._attention = None
        pass

    def forward(self, x):
        key = self._key_func(x)
        query = self._query_func(x)
        value = self._value_func(x)
        remain_dims, follow_dims = _functional._to_mat_dim_split(self._dims, len(key.shape))
        self._attention = functional.correlated(key, query, self._dims)
        right_value, _, _ = _functional._get_mat_right(value, self._dims)
        _out = torch.matmul(self._attention, right_value)
        out = _functional._get_tensor_right(_out, value.shape, self._dims)
        return out


    @property
    def attention(self):
        return self._attention
    pass