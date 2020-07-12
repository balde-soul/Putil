# coding=utf-8
#In[]
import Putil.data.coco as coco
from Putil.torch.loss.vision.ObjectDetection.center_net_loss import CenterNetLoss
from Putil.data.vision_common_convert.bbox_convertor import BBoxConvertToCenterBox as Convertor
import torch 
import numpy as np

coco.COCOData.set_seed(64)

loss = CenterNetLoss(0.5, 0.5)
box_net_out = torch.from_numpy(np.reshape(np.random.sample(int(81920 * 2 / 16)), [2, 5, 32, 32]).astype(np.float32))
print(box_net_out.shape)
class_net_out = torch.from_numpy(np.reshape(np.random.sample(int(1310720 * 2 / 16)), [2, 80, 32, 32]).astype(np.float32))
print(class_net_out.shape)

coco_data = coco.COCOData('/data2/Public_Data/COCO', coco.COCOData.Stage.STAGE_EVAL, '', detection=True)
convertor = Convertor(4, 80, sigma=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32))
coco_data.set_convert_to_input_method(convertor)
img_1, box_label_1, class_label_1, radiance_factor_1 = coco_data[1]
img_2, box_label_2, class_label_2, radiance_factor_2 = coco_data[2]
img = np.stack([img_1, img_2], axis=0)
box_label = np.stack([box_label_1, box_label_2], axis=0)
class_label = np.stack([class_label_1, class_label_2], axis=0)
radiance_factor = np.stack([radiance_factor_1, radiance_factor_2], axis=0)
print(box_label.shape)
print('not zero: {0}'.format(np.count_nonzero(box_label[:, :, 1: 3])))

print('loss: {0}'.format(loss(box_net_out.cuda(), class_net_out.cuda(), torch.from_numpy(box_label).cuda(), torch.from_numpy(class_label).cuda(), torch.from_numpy(radiance_factor).cuda())))