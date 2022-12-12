# coding=utf-8
from abc import ABCMeta, abstractmethod


class HistoryData:
    def __init__(self):
        pass
    pass

##@brief
# @note
# @time 2022-11-17
class SequentialHistoryDataCollector:
    def __init__(self, *args, **kwargs):
        pass

    ##@brief
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2022-11-21
    @abstractmethod
    def push(self, id, time, data, *args, **kwargs):
        pass

    @abstractmethod
    def get(self, id, s, e, *args, **kwargs):
        pass

    @abstractmethod
    def pop(self, id, s, e, *args, **kwargs):
        pass
    pass

##@brief
# @note
# @time 2022-11-17
class SequentialHistoryMetaDataCollector(SequentialHistoryDataCollector):
    pass