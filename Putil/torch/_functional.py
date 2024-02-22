# coding=utf-8
from functools import reduce
import torch
from Putil.base import putil_status


##@brief
# @note
# @param[in] dims
# @param[in] dimension
# return (remain_dims, follow_dims)
#   remain_dims:
#   follow_dims:
def _to_mat_dim_split(dims, dimension):
    temp = list(range(0, dimension))
    temp1 = list()
    for dim in dims:
        temp.remove(dim % dimension)
        temp1.append(dim % dimension)
    return temp, sorted(temp1)


def _get_tensor_right(mat, original_shape, dims):
    remain_dims, follow_dims = _to_mat_dim_split(dims, len(original_shape))

    _to_mat_permute = remain_dims[0: 1] + follow_dims + remain_dims[1: ] if len(remain_dims) > 1 else []
    permute = [s[1] for s in sorted(zip(_to_mat_permute, list(range(0, len(original_shape)))), key=lambda x: x[0])]

    mat = mat.view(_dims_to_shape(original_shape, remain_dims[0: 1]) + _dims_to_shape(original_shape, follow_dims) + _dims_to_shape(original_shape, remain_dims[1: ] if len(remain_dims) > 1 else []))
    #print(mat.shape) if putil_status.putil_is_debug() else None
    #print(mat) if putil_status.putil_is_debug() else None
    mat = mat.permute(permute).contiguous()
    #print(mat.shape) if putil_status.putil_is_debug() else None
    #print(mat) if putil_status.putil_is_debug() else None
    return mat


def _get_tensor_left(mat, original_shape, dims):
    remain_dims, follow_dims = _to_mat_dim_split(dims, len(original_shape))

    _to_mat_permute = remain_dims + follow_dims
    permute = [s[1] for s in sorted(zip(_to_mat_permute, list(range(0, len(original_shape)))), key=lambda x: x[0])]

    mat = mat.view(_dims_to_shape(original_shape, remain_dims[0: 1]) + _dims_to_shape(original_shape, remain_dims[1: ] if len(remain_dims) > 1 else []) + _dims_to_shape(original_shape, follow_dims))
    mat = mat.permute(permute).contiguous()
    return mat


def _dims_to_shape(shape, dims):
    return [shape[dim] for dim in dims]


def _resum_dims(shape, dims):
    return [reduce(lambda x, y: x * y, _dims_to_shape(shape, dims))]


def _get_mat_right(x, dims):
    shape = x.shape
    remain_dims, follow_dims = _to_mat_dim_split(dims, len(x.shape))
    follow_dim = [reduce(lambda x, y: x * y, _dims_to_shape(shape, follow_dims))]
    remain_dim_beside_batch = [reduce(lambda x, y: x * y, _dims_to_shape(shape, remain_dims[1: ]))] if len(remain_dims) > 1 else [1]
    x = x.permute(remain_dims[0: 1] + follow_dims + (remain_dims[1: ] if len(remain_dims) > 1 else [])).contiguous()
    #x = x.view(_dims_to_shape(shape, remain_dims[0: 1]) + follow_dim + _dims_to_shape(shape, remain_dims[1: ]) if len(remain_dims) > 1 else [])
    x = x.view(_dims_to_shape(shape, remain_dims[0: 1]) + follow_dim + remain_dim_beside_batch)
    return x, remain_dims, follow_dims


def _get_mat_left(x, dims):
    shape = x.shape
    remain_dims, follow_dims = _to_mat_dim_split(dims, len(x.shape))
    follow_dim = [reduce(lambda x, y: x * y, [x.shape[dim] for dim in follow_dims])]
    remain_dim_beside_batch = [reduce(lambda x, y: x * y, [x.shape[dim] for dim in remain_dims[1: ]])] if len(remain_dims) > 1 else [1]
    x = x.permute(remain_dims + follow_dims).contiguous()
    x = x.view(_dims_to_shape(shape, remain_dims[0: 1]) + remain_dim_beside_batch + follow_dim)
    return x, remain_dims, follow_dims