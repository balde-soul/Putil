# coding=utf-8
import numpy as np
import torch
from torch.nn import Module
import Putil.base.logger as plog

root_logger = plog.PutilLogConfig('CenterNetLoss').logger()
root_logger.setLevel(plog.DEBUG)
CenterNetLossLogger = root_logger.getChild('CenterNetLoss')
CenterNetLossLogger.setLevel(plog.DEBUG)

class CenterNetLoss(Module):
    '''
     @brief 
     @note
     this class provide a method to generate the center net object loss, the loss_obj, loss_xyshift, loss_wh
     loss_obj: represent whether obj localed in the cell or not
     loss_xyshift: work in the obj owned cell
     loss_wh: work in the obj owned cell
    '''
    def __init__(self, focal_alpha, decay_beta, class_weight=None, obj_loss_weight=0.3, offset_loss_weight=0.4, size_weight_loss=0.3):
        Module.__init__(self)
        self._focal_alpha = focal_alpha
        self._decay_beta = decay_beta
        self._class_weight = class_weight
        self._object_loss = None
        self._offset_loss = None

        self._obj_loss_weight = obj_loss_weight
        self._offset_loss_weight = offset_loss_weight
        self._size_weight_loss = size_weight_loss
        
        self._class_loss = torch.nn.CrossEntropyLoss(class_weight)
        pass

    def forward(self, box_net_out, class_net_out, box_label, class_label, radiance_factor):
        '''
         @brief 
         @note
         @param[in] box_net_out
         shape: [batch, 5, h, w], five dimension represent:[obj, x_shift, y_shift, w, h]
         @param[in] box_label
         shape: [batch, 6, h, w], five dimension represent: [obj, x_shift, y_shift, w, h]
         @param[in] class_label
         shape: [batch, h, w], label use in the CrossEntropyLoss
         @param[in] radiance_factor
         shape: [batch, 1, h, w], represent the
        '''
        #CenterNetLossLogger.debug(torch.nn.Conv2d.__doc__)
        #CenterNetLossLogger.debug('Transpose: \n {0}'.format(torch.transpose.__doc__))
        #CenterNetLossLogger.debug(box_net_out[0, 0, :, :].shape)
        obj = box_label[:, 0: 1, :, :]
        n_obj = 1 - box_label[:, 0: 1, :, :]
        obj_cell_amount = torch.nonzero(obj).size(0)
        n_obj_cell_amount = torch.nonzero(n_obj).size(0)
        p_obj_loss = torch.pow(1 - box_net_out[:, 0: 1, :, :], self._focal_alpha) * \
            box_label[:, 0: 1, :, :] * torch.log(box_net_out[:, 0: 1, :, :])
        n_obj_loss = torch.pow(1 - radiance_factor, self._decay_beta) * \
            torch.pow(box_net_out[:, 0: 1, :, :], self._focal_alpha) \
            * (1 - box_label[:, 0: 1, :, :]) * torch.log(1 - box_net_out[:, 0: 1, :, :])
        obj_loss = -torch.mean((obj_cell_amount / (obj_cell_amount + n_obj_cell_amount)) * p_obj_loss + \
            (n_obj_cell_amount / obj_cell_amount + n_obj_cell_amount) * n_obj_loss)
        CenterNetLossLogger.debug('obj_loss: {0}'.format(obj_loss))

        offset_loss = 1.0 / obj_cell_amount \
            * torch.sum(box_label[:, 0: 1, :, :] * \
                torch.abs(box_label[:, 1: 3, :, :] - box_net_out[:, 1: 3, :, :]))
        CenterNetLossLogger.debug('offset_loss: {0}'.format(offset_loss))

        wh_loss = 1.0 / obj_cell_amount \
            * torch.sum(box_label[:, 0: 1, :, :] * \
                torch.abs(box_label[:, 3: 5, :, :] - box_net_out[:, 3: 5, :, :]))
        CenterNetLossLogger.debug('wh_loss: {0}'.format(wh_loss))

        class_loss = self._class_loss(class_net_out, class_label)
        #return self._obj_loss_weight * obj_loss + self._offset_loss_weight * offset_loss + self._size_weight_loss * wh_loss + class_loss
        return self._offset_loss_weight * offset_loss + self._size_weight_loss * wh_loss + class_loss
    pass
##In[]:
#import torch
#CenterNetLossLogger.debug(torch.ones(1, 3, 2, 2).reshape((3, -1)).shape)