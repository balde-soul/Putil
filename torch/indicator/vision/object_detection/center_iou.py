# coding=utf-8

import torch
from torch.nn import Module
from Putil.torch.indicator.vision.object_detection.giou import GIoU


class CenterNetIoU(Module):
    def __init__(self, sample_rate):
        Module.__init__(self)
        self._sample_rate = sample_rate
        self._iou = GIoU()
        pass

    def forward(self, box_pre, box_gt, obj_gt):
        '''
        所有的gt-box都平移到top-left为[0， 0]处计算，使用偏移进行计算
        '''
        iou, giou = self._iou(box_pre * self._sample_rate, box_gt * self._sample_rate)
        giou = torch.sum(giou * obj_gt) / torch.sum(obj_gt)
        return giou