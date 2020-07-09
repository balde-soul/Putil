# coding=utf-8
#In[]:
import numpy as np
import torch
from torch.nn import Module

class CenterNetLoss(Module):
    def __init__(self):
        Module.__init__(self)
        self._object_loss = None
        self._offset_loss = None
        pass

    def forward(self, net_out, label):
        return 0
    pass

import Putil.data.coco as coco
coco.COCOData.set_seed(64)

loss = CenterNetLoss()
net_out = torch.from_numpy(np.reshape(np.random.sample(81920), [1, 5, 128, 128]))
print(net_out.shape)

coco_data = coco.COCOData('/data2/Public_Data/COCO', coco.COCOData.Stage.STAGE_EVAL, '', detection=True)
convertor = Convertor(4, sigma=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32))
coco_data.set_convert_to_input_method(convertor)
img, label = coco_data[1]

loss(net_out, label)