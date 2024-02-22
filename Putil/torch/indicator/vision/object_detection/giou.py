# coding=utf-8

import torch
from torch.nn import Module
from Putil.torch.indicator.vision.object_detection import box
from Putil.torch.indicator.vision.object_detection import iou

def _argmin_area(x11, y11, x12, y12, x21, y21, x22, y22):
    cx1, cy1, cx2, cy2 = box._argmin_box(x11, y11, x12, y12, x21, y21, x22, y22)
    area_c = box._box_area(cx1, cy1, cx2, cy2)
    return area_c


##@brief
# @note
# https://arxiv.org/pdf/1902.09630.pdf
# C=\underset{C}{argmin}{(pre\cup gt\subseteq C)} 
# GIoU=IoU-\frac{{{\parallel C -(pre\cup gt)\parallel}_0}}{{\parallel C\parallel}_0}
# iou在iou为0时无法直接优化，针对bbox的回归无法等价于iou的回归，提出giou可作为目标进行优化
# @param[in] pre
# float or double, positivated, [batch_size, one_box, ...] content of box: 
# (top_left_x + any_x_shift, top_left_y + any_y_shift, width, height)
# @prarm[in] gt 
# float or double, positivated, [batch_size, one_box, ...] content of box: 
# (top_left_x + any_x_shift, top_left_y + any_y_shift, width, height)
# @ret
# 0: the iou [batch_size, ..., 1]
# 1: the giou [batch_size, ..., 1]
class GIoU(iou.iou):
    def iou_index(self):
        return 1 

    def __init__(self):
        iou.iou.__init__(self)
        pass

    def forward(self, box1, box2):
        box1 = box._tlwh_to_tlbr(box1)
        box2 = box._tlwh_to_tlbr(box2)
        x11, y11, x12, y12 = box._to_xyxy(box1)
        x21, y21, x22, y22 = box._to_xyxy(box2)

        cap, cup = box._cap_cup(x11, y11, x12, y12, x21, y21, x22, y22)

        _iou = iou._cap_cup_iou(cap, cup)

        _area_c = _argmin_area(x11, y11, x12, y12, x21, y21, x22, y22)

        _giou = _iou - ((_area_c - cup) / _area_c + 1e-32)
        return _iou, _giou