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

class iou(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        pass

    ##@brief 返回当前对象的准确iou值索引，有些的返回值可能有多个数据（包含过程数据以及基础iou等），需要该接口方便的返回对应iou的索引
    # @return int 索引
    @abstractmethod
    def iou_index(self):
        pass

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