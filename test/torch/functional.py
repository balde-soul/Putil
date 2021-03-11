# coding=utf-8
import torch
from Putil.base import putil_status
from Putil.torch.functional import *
import time

putil_status.set_putil_mode(putil_status.PutilMode.Debug)
size = [1, 2, 3, 4]
data = torch.rand(size)
data2 = torch.rand(size)
data3 = torch.rand(size)

if putil_status.putil_is_debug():
    from Putil.torch._functional import _get_mat_left, _get_mat_right, _get_tensor_left, _get_tensor_right
    dims = [[1, 3], [2, 3], [1, 2]]
    print(data.shape)
    print(data)
    for _dims in dims:
        print('check get_mat<->get_tensor: {}'.format(_dims))
        _data, _, _ = _get_mat_left(data, _dims)
        print(_data.shape)
        print(_data)
        _data = _get_tensor_left(_data, list(data.shape), _dims)
        print('_data is equal data: {}'.format((_data - data).gt(0.).sum() == 0))
        print(_data.shape)
        print(_data)
        _data, _, _ = _get_mat_right(data, _dims)
        print(_data.shape)
        print(_data)
        _data = _get_tensor_right(_data, list(data.shape), _dims)
        print('_data is equal data: {}'.format((_data - data).gt(0.).sum() == 0))
        print(_data.shape)
        print(_data)
    pass

if putil_status.putil_is_debug():
    print('origin test view')
    print(data.shape)
    print(data)
    print('permute->view')
    _data = data.permute([0, 2, 3, 1]).contiguous()
    print(_data.shape)
    print(_data)
    _data = _data.view([1, 12, 2])
    print(_data.shape)
    print(_data)
    print('permute->view back')
    _data = _data.view([1, 3, 4, 2])
    print(_data.shape)
    print(_data)
    _data = _data.permute([0, 3, 1, 2]).contiguous()
    print(_data.shape)
    print(_data)
    print('_data is equal data: {}'.format((_data - data).gt(0.).sum() == 0))
    print('view->permute')
    _data = _data.view([1, 2, 12])
    print(_data.shape)
    print(_data)
    _data = _data.permute([0, 2, 1]).contiguous()
    print(_data.shape)
    print(_data)
    print('view->permute back')
    _data = _data.permute([0, 2, 1]).contiguous()
    print(_data.shape)
    print(_data)
    _data = _data.view([1, 2, 3, 4])
    print(_data.shape)
    print(_data)
    print('_data is equal data: {}'.format((_data - data).gt(0.).sum() == 0))
    pass

begin = time.time()
follow_dims = [2, 3]
corr = correlated(data, data2, follow_dims)
print('corr({0}, {3}) follow {4}->{1} cost: {2}'.format(data.shape, corr.shape, time.time() - begin, data2.shape, follow_dims))

begin = time.time()
follow_dims = [1]
corr = correlated(data, data2, follow_dims)
print('corr({0}, {3}) follow {4}->{1} cost: {2}'.format(data.shape, corr.shape, time.time() - begin, data2.shape, follow_dims))

#begin = time.clock()
#corr = correlated(data, data2, list(range(0, len(size)))[1: ])
#print('{0}->{1} cost: {2}'.format(data.shape, corr.shape, time.clock() - begin))

begin = time.time()
follow_dims = [2, 3]
weighted = correlated_weight(data, data2, data3, follow_dims)
print('corr({0}, {3}) weight {4} follow {5} ->{1} cost: {2}'.format(data.shape, weighted.shape, time.time() - begin, data2.shape, data3.shape, follow_dims))

begin = time.time()
follow_dims = [1]
weighted = correlated_weight(data, data2, data3, follow_dims)
print('corr({0}, {3}) weight {4} follow {5} ->{1} cost: {2}'.format(data.shape, weighted.shape, time.time() - begin, data2.shape, data3.shape, follow_dims))