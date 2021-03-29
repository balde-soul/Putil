# coding=utf-8

from enum import Enum
from abc import ABCMeta, abstractmethod

class BoxIoU(metaclass=ABCMeta):
    class IoUType(Enum):
        IoU = 0
        GIoU = 1
        DIoU = 2
        CIoU = 3

    def __init__(self, iou_type):
        self._iou_type = iou_type
        self._iou_calculator = None
        self._generate_iou_calculate()
        pass

    @abstractmethod
    def _generate_iou_calculate(self):
        pass