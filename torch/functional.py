# coding=utf-8
import torch
from Putil.base import putil_status
from Putil.torch._functional import _get_mat_right, _get_mat_left, _to_mat_dim_split, _dims_to_shape, _resum_dims, _get_tensor_left, _get_tensor_right


def correlated(x, y, dims):
    print(x.shape) if putil_status.putil_is_debug() else None
    print(x) if putil_status.putil_is_debug() else None
    print(y.shape) if putil_status.putil_is_debug() else None
    print(y) if putil_status.putil_is_debug() else None
    right, _, _ = _get_mat_right(x, dims)
    print(right.shape) if putil_status.putil_is_debug() else None
    print(right) if putil_status.putil_is_debug() else None
    left, _, _ = _get_mat_left(y, dims)
    print(left.shape) if putil_status.putil_is_debug() else None
    print(left) if putil_status.putil_is_debug() else None
    corr = torch.matmul(right, left)
    print(corr.shape) if putil_status.putil_is_debug() else None
    print(corr) if putil_status.putil_is_debug() else None
    return corr


def correlated_weight(corr_x, corr_y, value, dims):
    corr = correlated(corr_x, corr_y, dims)
    _value, _, _ = _get_mat_right(value, dims)
    print(value.shape) if putil_status.putil_is_debug() else None
    print(value) if putil_status.putil_is_debug() else None
    weighted = torch.matmul(corr, _value)
    weighted = torch.softmax(weighted, dim=-1)
    print(weighted.shape) if putil_status.putil_is_debug() else None
    print(weighted) if putil_status.putil_is_debug() else None
    weighted = _get_tensor_right(weighted, list(value.shape), dims)
    return weighted