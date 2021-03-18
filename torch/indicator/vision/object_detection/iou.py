# coding = utf-8
from abc import ABCMeta, abstractmethod
import torch
from Putil.torch.indicator.vision.object_detection import box

##@brief 计算iou
# @note
# @return 
def _iou(x11, y11, x12, y12, x21, y21, x22, y22):
    cap, cup = box._cap_cup(x11, y11, x12, y12, x21, y21, x22, y22)
    return cap / (cup + 1e-32)

def _cap_cup_iou(cap, cup):
    return cap / (cup + 1e-32)

##@brief 计算IoU，基于[batch, box, ...]进行计算，box的结构是[top_left_x, top_left_y, width, height], 
# 返回的是[batch, 1, ...]，第二维表示的是iou值，当前单元不存在gt_box的情况使用[0, 0, 0, 0]代表，
# 那么不同的iou，针对不存在gt的情况获得的值就不一样，需要特别注明 **一般情况下，计算一个batch的MeanIoU都是需要
# 进
# @note
class iou(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        pass

    ##@brief 返回当前对象的准确iou值索引，有些的返回值可能有多个数据（包含过程数据以及基础iou等），需要该接口方便的返回对应iou的索引
    # @return int 索引
    @abstractmethod
    def iou_index(self):
        pass

    @abstractmethod
    def iou_mean(self, iou):
        pass

class MeanIoU(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        pass

    def forward(self, iou, obj_gt):
        iou_filtered = iou * obj_gt
        iou = torch.nansum(iou_filtered) / (torch.isnan(iou_filtered).eq(False) * obj_gt).sum()
        return iou
        

##@brief
# @note
class IoU(iou):
    def iou_index(self):
        return 0

    def __init__(self):
        iou.__init__(self)
        pass

    def forward(self, box1, box2):
        box1 = box._tlwh_to_tlbr(box1)
        box2 = box._tlwh_to_tlbr(box2)
        x11, y11, x12, y12 = box._to_xyxy(box1)
        x21, y21, x22, y22 = box._to_xyxy(box2)
        iou = _iou(x11, y11, x12, y12, x21, y21, x22, y22)
        return iou, 