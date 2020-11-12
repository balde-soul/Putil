# coding=utf-8
from abc import abstractmethod
from torch.nn import Module


class EvaluateIndicator(Module):
    '''
     @brief
     @note 接收evaluate阶段的decode输出，生成evaluate数据集的目标指标
    '''
    def __init__(self, args):
        Module.__init__(self)
        self._evaluate_indicator_name = args.evaluate_indicator_name
        self._evaluate_indicator_source = args.evaluate_indicator_source
        pass

    @abstractmethod
    def add_step_output(self, *input):
        '''
         @brief
         @note 每个step的decode输出传入到该函数，进行统计处理
        '''
        pass

    @abstractmethod
    def statistic_out(self, save_to_file=None, *kargs, **kwargs):
        '''
         @brief
         @note 最后一个step运行完，decode输出传到add_step_output之后就可以调用此函数进行统计，结果输出同时保存到本身维持的一个文本
        '''
        pass
    pass
