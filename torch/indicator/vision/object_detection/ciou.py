# coding=utf-8

import numpy as np
import torch
from Putil.torch.indicator.vision.object_detection import box
from Putil.torch.indicator.vision.object_detection import iou
from Putil.torch.indicator.vision.object_detection import diou
from Putil.torch.indicator.vision.object_detection import giou

##@brief
# @note
class CIoU(iou.iou):
    def iou_index(self):
        return 0

    def __init__(self):
        iou.iou.__init__(self)
        self._square_pi = torch.nn.parameter.Parameter(torch.pow(torch.Tensor([np.pi]), 2))
        self.register_parameter('CIoU_pi', self._square_pi)
        pass

    def forward(self, box1, box2):
        x11, y11, x12, y12 = box._to_xyxy(box._tlwh_to_tlbr(box1))
        x21, y21, x22, y22 = box._to_xyxy(box._tlwh_to_tlbr(box2))
        cx1, cy1, width1, height1 = box._to_cxcywh(box._tlwh_to_cxcywh(box1))
        cx2, cy2, width2, height2 = box._to_cxcywh(box._tlwh_to_cxcywh(box2))
        _iou = iou._iou(x11, y11, x12, y12, x21, y21, x22, y22)
        _area_c = giou._argmin_area(x11, y11, x12, y12, x21, y21, x22, y22)
        _square_d = diou._square_center_distance(cx1, cy1, cx2, cy2)
        _v = 4 * torch.pow(torch.arctan(width1) / torch.arctan(height1) - torch.arctan(width2) / torch.arctan(height2), 2) / self._square_pi
        #with torch.no_grad():
        #    _alpha = _v / (1 - _iou + _v)
        _alpha = _v / (1 - _iou + _v)
        _ciou = _iou - _square_d / (torch.pow(_area_c, 2) + 1e-21) - _alpha * _v 
        return  _ciou, _iou, _area_c, _square_d, _v, _alpha