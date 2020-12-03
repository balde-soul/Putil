# coding=utf-8
from abc import ABCMeta, abstractmethod
from enum import Enum

import Putil.base.logger as plog

logger = plog.PutilLogConfig('data_sampler').logger()
logger.setLevel(plog.DEBUG)
torch_DataSampler_logger = logger.getChild('torch_DataSampler')
torch_DataSampler_logger.setLevel(plog.DEBUG)


class DataSampler(metaclass=ABCMeta):
    def __init__(self, args):
        self._args = args
        pass

    def __call__(self, dataset, rank_amount, rank):
        '''
         @brief generate the data sampler
         @param[in] dataset follow the common data
         @param[in] rank_amount the amount of the rank
         @param[in] rank the rank of this sampler
         @ret
         DataSampler
        '''
        pass


class DefaultDataSampler(DataSampler):
    def __init__(self, args):
        torch_DataSampler_logger.info('use torch DataSampler')
        DataSampler.__init__(self, args)
        from torch.utils.data.distributed import DistributedSampler as data_sampler
        pass

    def __call__(self, dataset, rank_amount, rank):
        return data_sampler(dataset, num_replicas=hvd.size(), rank=hvd.rank())
    pass


def DefaultDataSamplerArg(parser):
    pass