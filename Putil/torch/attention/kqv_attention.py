# coding=utf-8
import torch


## @brief
class KQV(torch.nn.Module):
    def __init__(self, dims, key_module, query_module, value_module):
        torch.nn.Module.__init__(self)
        assert len(dims) > 0
        self._dims = dims
        self._Key = key_module
        self._Query = query_module
        self._Value = value_module
        pass

    ##@brief 
    # @param[in] x shape:[Batch, *]如果Batch为1则需要为[1, *]
    def forward(self, x):
        remain_dims, correlated_dims = get_KQV_dim(self._dims, len(x.shape))
        key = self._Key(x)
        # the key is on [batch, remain_dims_sum/1, correlated_dims_sum]
        key = key.permute(remain_dims + correlated_dims).contiguous().view(remain_dims + [-1]).view([remain_dims[0]] + -1 if len(remain_dims) > 1 else [1] + sum(correlated_dims))
        query = self._Query(x)
        query = query.permute(remain_dims[0: 1] + correlated_dims + remain_dims[1: ] if len(remain_dims) > 1 else []).contiguous().view(remain_dims[0: 1] + -1 + remain_dims[1: ] if len(remain_dims) > 1 else []).view(remain_dims[0: 1] + sum(correlated_dims) + [-1] if len(remain_dims) > 1 else [1])
        value = self._Value(x)
        value = value.permute(
        self_attention = torch.matmul(query, key.transpose(1, 0))
        output = torch.matmul(self_attention, value)
        return key, query, value, attention, output
    pass