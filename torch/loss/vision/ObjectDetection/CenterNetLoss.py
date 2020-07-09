# coding=utf-8
#In[]:
import numpy as np
import torch
from torch.nn import Module

class CenterNetLoss(Module):
    def __init__(self, focal_alpha, decay_beta, obj_loss_weight=0.3, offset_loss_weight=0.4, size_weight_loss=0.3):
        Module.__init__(self)
        self._focal_alpha = focal_alpha
        self._decay_beta = decay_beta
        self._object_loss = None
        self._offset_loss = None

        self._obj_loss_weight = obj_loss_weight
        self._offset_loss_weight = offset_loss_weight
        self._size_weight_loss = size_weight_loss
        pass

    def forward(self, net_out, label):
        #print(torch.nn.Conv2d.__doc__)
        #print('Transpose: \n {0}'.format(torch.transpose.__doc__))
        print(net_out[0, 0, :, :].shape)
        p_obj_loss = torch.pow(1 - net_out[0, 0, :, :], self._focal_alpha) * label[:, :, 0] * torch.log(net_out[0, 0, :, :])
        n_obj_loss = torch.pow(1 - label[:, :, 6], self._decay_beta) * torch.pow(net_out[0, 0, :, :], self._focal_alpha) * (1 - label[:, :, 0]) * torch.log(net_out[0, 0, :, :])
        obj_loss = -torch.mean(p_obj_loss + n_obj_loss)
        print('obj_loss: {0}'.format(obj_loss))
        offset_loss = 1.0 / torch.nonzero(label[:, :, 0]).size(0) \
            * torch.sum(label[:, :, 0] * torch.abs( \
                torch.transpose(torch.transpose(label[:, :, 1: 3], -1, 1), 0, 1)  - net_out[0, 1: 3, :, :]))
        print('offset_loss: {0}'.format(offset_loss))
        wh_loss = 1.0 / torch.nonzero(label[:, :, 0]).size(0) \
            * torch.sum(label[:, :, 0] * torch.abs( \
                torch.transpose(torch.transpose(label[:, :, 3: 5], -1, 1), 0, 1)  - net_out[0, 3: 5, :, :]))
        print('wh_loss: {0}'.format(wh_loss))
        return self._obj_loss_weight * obj_loss + self._offset_loss_weight * offset_loss + self._size_weight_loss * wh_loss
    pass

import Putil.data.coco as coco
coco.COCOData.set_seed(64)

loss = CenterNetLoss(0.5, 0.5)
net_out = torch.from_numpy(np.reshape(np.random.sample(81920), [1, 5, 128, 128]).astype(np.float32))
print(net_out.shape)

coco_data = coco.COCOData('/data2/Public_Data/COCO', coco.COCOData.Stage.STAGE_EVAL, '', detection=True)
from Putil.data.vision_common_convert.bbox_convertor import BBoxConvertToCenterBox as Convertor
convertor = Convertor(4, sigma=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32))
coco_data.set_convert_to_input_method(convertor)
img, label = coco_data[1]
print(label.shape)
print('not zero: {0}'.format(np.count_nonzero(label[:, :, 1: 3])))

loss(net_out, torch.from_numpy(label))
