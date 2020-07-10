# coding=utf-8
#In[]:
import numpy as np
import torch
from torch.nn import Module

class CenterNetLoss(Module):
    '''
     @brief 
     @note
     this class provide a method to generate the center net object loss, the loss_obj, loss_xyshift, loss_wh
     loss_obj: represent whether obj localed in the cell or not
     loss_xyshift: work in the obj owned cell
     loss_wh: work in the obj owned cell
    '''
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
        '''
         @brief 
         @note
         @param[in] net_out
         shape: [batch, 5, h, w], five dimension represent:[obj, x_shift, y_shift, w, h]
         @param[in] label
         shape: [batch, h, w, 5], five dimension represent: [obj, x_shift, y_shift, w, h]
        '''
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