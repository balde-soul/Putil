# coding=utf-8
#In[]
import Putil.data.coco as coco
from Putil.torch.loss.vision.ObjectDetection.CenterNetLoss import CenterNetLoss
from Putil.data.vision_common_convert.bbox_convertor import BBoxConvertToCenterBox as Convertor
import torch 
import numpy as np

coco.COCOData.set_seed(64)

loss = CenterNetLoss(0.5, 0.5)
net_out = torch.from_numpy(np.reshape(np.random.sample(81920), [1, 5, 128, 128]).astype(np.float32))
print(net_out.shape)

coco_data = coco.COCOData('/data2/Public_Data/COCO', coco.COCOData.Stage.STAGE_EVAL, '', detection=True)
convertor = Convertor(4, sigma=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32))
coco_data.set_convert_to_input_method(convertor)
img, label = coco_data[1]
print(label.shape)
print('not zero: {0}'.format(np.count_nonzero(label[:, :, 1: 3])))

loss(net_out, torch.from_numpy(label))