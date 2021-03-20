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


##@brief 将通过_get_mat_right的结果转换会去
# @note 
# @param[in] mat 输入_get_mat_right的结果
# @param[in] original_shape 经过_get_mat_right之前的tensor的shape
# @param[in] dims 输入_get_mat_right的dims
# @return torch.tensor
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


##@brief 将通过_get_mat_left的结果转换会去
# @note 
# @param[in] mat 输入_get_mat_left的结果
# @param[in] original_shape 经过_get_mat_left之前的tensor的shape
# @param[in] dims 输入_get_mat_left的dims
# @return torch.tensor
def _get_tensor_left(mat, original_shape, dims):
    remain_dims, follow_dims = _to_mat_dim_split(dims, len(original_shape))

    _to_mat_permute = remain_dims + follow_dims
    permute = [s[1] for s in sorted(zip(_to_mat_permute, list(range(0, len(original_shape)))), key=lambda x: x[0])]

    mat = mat.view(_dims_to_shape(original_shape, remain_dims[0: 1]) + _dims_to_shape(original_shape, remain_dims[1: ] if len(remain_dims) > 1 else []) + _dims_to_shape(original_shape, follow_dims))
    mat = mat.permute(permute).contiguous()
    return mat

##@brief 将矩阵乘的结果变换成目标结果
# @note
# @param[in]
# @param[in]
# @return 
def _get_tensor_from_matmul(mat, left_dims, right_dims, left_shape, right_shape):
    left_remain_dim, left_follow_dim = _to_mat_dim_split(left_dims, len(left_shape))
    right_remain_dim, right_follow_dim = _to_mat_dim_split(right_dims, len(right_shape))
    mat = mat.view(left_shape[0: 1] + _dims_to_shape(left_shape, left_remain_dim) + _dims_to_shape(right_shape, right_remain_dim))
    return mat


def _dims_to_shape(shape, dims):
    return [shape[dim] for dim in dims]


def _resum_dims(shape, dims):
    return [reduce(lambda x, y: x * y, _dims_to_shape(shape, dims))]


##@brief 将tensor变换成[batch, row, col]的格式，让其支持matmul, 右乘 乘子
# @note 变换规则，指定dims，该对应的dims会被置于row，row是dims的累乘，col为除去dims与batch维度的其他维度的累乘
# @param[in] x tensor需要batch在第一维度，shape为[batch, ...]
# @param[in] dims list 指定的维度将作为col
# @return tensor 支持matmul的格式，shape为[batch, row, col]
def _get_mat_right(x, dims):
    shape = x.shape
    remain_dims, follow_dims = _to_mat_dim_split(dims, len(x.shape))
    follow_dim = [reduce(lambda x, y: x * y, _dims_to_shape(shape, follow_dims))]
    remain_dim_beside_batch = [reduce(lambda x, y: x * y, _dims_to_shape(shape, remain_dims[1: ]))] if len(remain_dims) > 1 else [1]
    x = x.permute(remain_dims[0: 1] + follow_dims + (remain_dims[1: ] if len(remain_dims) > 1 else [])).contiguous()
    #x = x.view(_dims_to_shape(shape, remain_dims[0: 1]) + follow_dim + _dims_to_shape(shape, remain_dims[1: ]) if len(remain_dims) > 1 else [])
    x = x.view(_dims_to_shape(shape, remain_dims[0: 1]) + follow_dim + remain_dim_beside_batch)
    return x, remain_dims, follow_dims


##@brief 将tensor变换成[batch, row, col]的格式，让其支持matmul 左乘 乘子
# @note 变换规则，指定dims，该对应的dims会被置于col，col是dims的累乘，row为除去dims与batch维度的其他维度的累乘
# @param[in] x tensor需要batch在第一维度，shape为[batch, ...]
# @param[in] dims list 指定的维度将作为col
# @return tensor 支持matmul的格式，shape为[batch, row, col]
def _get_mat_left(x, dims):
    shape = x.shape
    remain_dims, follow_dims = _to_mat_dim_split(dims, len(x.shape))
    follow_dim = [reduce(lambda x, y: x * y, [x.shape[dim] for dim in follow_dims])]
    remain_dim_beside_batch = [reduce(lambda x, y: x * y, [x.shape[dim] for dim in remain_dims[1: ]])] if len(remain_dims) > 1 else [1]
    x = x.permute(remain_dims + follow_dims).contiguous()
    x = x.view(_dims_to_shape(shape, remain_dims[0: 1]) + remain_dim_beside_batch + follow_dim)
    return x, remain_dims, follow_dims