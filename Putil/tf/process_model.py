# coding=utf-8

"""
this file offer some method to process a serials of model
boost , compare, ...,etc
"""
import Putil.loger as plog

root_logger = plog.PutilLogConfig("ProcessModel").logger()
root_logger.setLevel(plog.DEBUG)
AvgModelLogger = root_logger.getChild("AvgModelLogger")
AvgModelLogger.setLevel(plog.DEBUG)


class AvgModel:
    def __init__(self):
        pass

    def __avg_model(self):
        return 0
        pass

    def AvgModel(self):
        return self.__avg_model()
        pass
    pass
