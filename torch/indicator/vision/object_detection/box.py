# coding=utf-8
import copy
import torch
# tlwh: boxes, shape[batch, 4, ...], box format: (top_left_x, top_left_y, width, height)
# tlbr: boxes, shape[batch, 4, ...], box format: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
# cxcywh: boxes, shape[batch, 4, ...], box format: (center_x, center_y, width, height)

##@brief
# @note
# @param[in] boxes, shape[batch, 4, ...], box format: (top_left_x, top_left_y, width, height)
# @return cxcywh: boxes, shape[batch, 4, ...], box format: (center_x, center_y, width, height)
def _tlwh_to_cxcywh(box):
    box = torch.cat([box[:, 0: 2], box[:, 0: 2] + box[:, 2: 4]], dim=1)
    return box

##@brief tlwh 模式的box转为tlbr
# @note
# @param[in] boxes, shape[batch, 4, ...], box format: (top_left_x, top_left_y, width, height)
# @return tlbr: boxes, shape[batch, 4, ...], box format: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
def _tlwh_to_tlbr(box):
    box = torch.cat([box[:, 0: 2], box[:, 0: 2] + box[:, 2: 4]], dim=1)
    return box

##@brief
# @note
# @param[in] boxes, shape[batch, 4, ...], box can be the output of _tlwh_to_tlbr with format: (top_left_x, top_left_y, width, height)
# @return top_left_x, top_left_y, width, height, shape[batch， 1， ...] for every element
def _to_xywh(box):
    return box[:, 0: 1], box[: 1: 2], box[:, 2: 3], box[:, 3: 4]

##@brief 
# @note
# @param[in] boxes, shape[batch, 4, ...], box can be the output of _tlwh_to_tlbr with format: (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
# @return top_left_x, top_left_y, bottom_right_x, bottom_right_y, shape[batch， 1， ...] for every element
def _to_xyxy(box):
    return box[:, 0: 1], box[:, 1: 2], box[:, 2: 3], box[:, 3: 4]

##@brief 
# @note
# @param[in] boxes, shape[batch, 4, ...], box can be the output of _tlwh_to_cxcywh with format: (center_x, center_y, bottom_right_x, bottom_right_y)
# @return center_x, center_y, bottom_right_x, bottom_right_y, shape[batch， 1， ...] for every element
def _to_cxcywh(box):
    return box[:, 0: 1], box[: 1: 2], box[:, 2: 3], box[:, 3: 4]

##@brief 计算box的面积
# @note
# @param[in]
# @param[in]
# @param[in]
# @param[in]
# @return 
def _box_area(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

##@brief 获取cap的box结构
# @note
# @return
def _cap_box(x11, y11, x12, y12, x21, y21, x22, y22):
    cap_x1 = torch.max(x11, x21)
    cap_y1 = torch.max(y11, y21)
    cap_x2 = torch.min(x12, x22)
    cap_y2 = torch.min(y12, y22)
    return cap_x1, cap_y1, cap_x2, cap_y2

##@brief 获取最小包含两个box的正矩形
# @note
# @return 
def _argmin_box(x11, y11, x12, y12, x21, y21, x22, y22):
    closure_x1 = torch.min(x11, x21)
    closure_y1 = torch.min(y11, y21)
    closure_x2 = torch.max(x12, x22)
    closure_y2 = torch.max(y12, y22)
    return closure_x1, closure_y1, closure_x2, closure_y2

##@brief 获取cap的面积
# @note
# @return 
def _cap(x11, y11, x12, y12, x21, y21, x22, y22):
    cap_x1, cap_y1, cap_x2 ,cap_y2 = _cap_box(x11, y11, x12, y12, x21, y21, x22, y22)
    mask = (cap_y2 > cap_y1) * (cap_x2 > cap_x1)
    cap = (cap_x2 * mask - cap_x1 * mask) * (cap_y2 * mask - cap_y1 * mask) # cap 
    return cap

##@brief 获取cup的面积
# @note
# @return 
def _cup(x11, y11, x12, y12, x21, y21, x22, y22, cap):
    cup = (x12 - x11) * (y12 - y11) + (x22 - x21) * (y22 - y21) - cap
    return cup

##@brief 获取cap和cup的面积
# @note 为什么要有这个函数呢？是因为cap和cup运算有重复块，这是避免浪费时间
# @return 
def _cap_cup(x11, y11, x12, y12, x21, y21, x22, y22):
    cap = _cap(x11, y11, x12, y12, x21, y21, x22, y22)
    cup = _cup(x11, y11, x12, y12, x21, y21, x22, y22, cap)
    return cap, cup