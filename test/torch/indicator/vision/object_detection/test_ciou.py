# coding=utf-8
#In[]
import os
from Putil.base import jupyter_state
jupyter_state.go_to_top(6, os.path.abspath(__file__))

import torch
import numpy as np
import random

from Putil.torch.util import set_torch_deterministic as STD
from Putil.test.torch.indicator.vision.object_detection import test_iou

from Putil.torch.indicator.vision.object_detection.ciou import CIoU as IoU

iou = IoU()
iou_data = test_iou.test_iou(iou)
print('min: {}'.format(iou_data[iou.iou_index()].min()))
print('max: {}'.format(iou_data[iou.iou_index()].max()))
