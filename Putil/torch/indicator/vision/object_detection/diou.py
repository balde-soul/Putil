# coding=utf-8

import torch 
from Putil.torch.indicator.vision.object_detection import box
from Putil.torch.indicator.vision.object_detection import iou

def _square_center_distance(cx1, cy1, cx2, cy2):
    return torch.pow(cx2 - cx1, 2) + torch.pow(cy2 - cy1, 2)


##@brief
# @note
# https://arxiv.org/pdf/1911.08287.pdf
# DIoU = IoU - \
# @param[in] pre
# float or double, positivated, [batch_size, ..., one_box] content of box: (top_left_x, top_left_y, width, height)
# @prarm[in] gt 
# float or double, positivated, [batch_size, ..., one_box] content of box: (top_left_x, top_left_y, width, height)
# @ret
# 0: the iou [batch_size, ..., 1]
# 1: the giou [batch_size, ..., 1]
class DIoU(iou.iou):
    def iou_index(self):
        return 0

    def __init__(self):
        iou.iou.__init__(self)
        pass

    def forward(self, box1, box2):
        cxcywh1 = box._tlwh_to_cxcywh(box1)
        cx1, cy1, _, _ = box._to_cxcywh(cxcywh1)
        cxcywh2 = box._tlwh_to_cxcywh(box2)
        cx2, cy2, _, _ = box._to_cxcywh(cxcywh2)

        tlbr1 = box._tlwh_to_tlbr(box1)
        tlbr2 = box._tlwh_to_tlbr(box2)
        x11, y11, x12, y12 = box._to_xyxy(tlbr1)
        x21, y21, x22, y22 = box._to_xyxy(tlbr2)

        _iou = iou._iou(x11, y11, x12, y12, x21, y21, x22, y22)

        cbox_x1, cbox_y1, cbox_x2, cbox_y2 = box._argmin_box(x11, y11, x12, y12, x21, y21, x22, y22)
        _area_c = box._box_area(cbox_x1, cbox_y1, cbox_x2, cbox_y2)
        _square_d = _square_center_distance(cx1, cy1, cx2, cy2)

        diou = _iou - _square_d / (torch.pow(_area_c, 2) + 1e-32)
        return diou, _iou, _area_c, _square_d