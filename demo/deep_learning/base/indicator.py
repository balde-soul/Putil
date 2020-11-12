'''
 @note
 本文件提供指标计算方法，该指标应用于train与evaluate中的量化，通过该指标判断train与evaluate的基本情况
 以及进行evaluate是否提升的判断
'''
# coding=utf-8
from abc import abstractmethod
from torch.nn import Module


class Indicator(Module):
    '''
     @brief
     @note 每个epoch一个轮回，接收train阶段evaluate的decode输出，生成代表性指标，指导lr_reduce、auto_save、auto_stop等
    '''
    def __init__(self, args):
        Module.__init__(self)
        self._indicator_name = args.indicator_name
        self._indicator_source = args.indicator_source
        self._save_dir = args.save_dir
        self._indicators = dict()
        self._interval_indicators = dict()
        self._epoch_indicator_mean = dict()
        self._epoch = 0
        pass

    @abstractmethod
    def forward(self, *input):
        '''
         @brief
         @note 每个step的模型输出与数据传入该函数
        '''
        pass

    @abstractmethod
    def combine_indicator(self):
        '''
         @brief
         @note train阶段的evalute时段整个epoch结束之后调用该函数，输出一个指标对象，该对象支持<比较函数>
        '''
        pass

    def set_epoch(self, epoch):
        '''
         @brief
         @note 调用此函数，更新状态，每个epoch开始都需要调用
        '''
        self._epoch = epoch
        self._indicators = dict()
        pass

    def get_indicators(self, epoch):
        pass