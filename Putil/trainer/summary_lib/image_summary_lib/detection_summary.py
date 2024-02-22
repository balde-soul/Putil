# coding=utf-8
import tensorboardX

import numpy as np
import torch
import cv2
from enum import Enum


class DetectionVisual:
    class BoundaryType(Enum):
        '''
        RegularRectangle in ObliqueRectangle in Polygon
        '''
        Polygon = 2
        ObliqueRectangle = 0
        RegularRectangle = 0 # [[x, y, w, h], ...] list or tensor
    
    def __init__(self, boundary_type, color_map=None, class_name=None):
        # TODO: check the boundary_type in the DetectionSummary.BoundaryType
        assert boundary_type in DetectionVisual.BoundaryType
        self._class_color_map = color_map
        self._class_name = class_name
        pass

    def draw(self, image, target_information, global_step):
        #if isinstance(image, np.ndarray) or isinstance(image, torch.Tensor):
        #    batch, channel, height, width = image.shape
        #if isinstance(image, list):
        #    batch = len(image)
        #    pass
        batch, channel, height, width = image.shape
        writer = tensorboardX.SummaryWriter('./')
        for b in range(0, batch):
            writer.add_image_with_boxes('', image[b], target_information[b], global_step=global_step, )
        pass


def general_rectangle_clamp(box_tensor, )


def torch_rectangle_image_summary(writer, tag, image_array, box_list, global_step, number=16, \
    color_index=None, color_map=None, batch_dim=0, dataformats='CHW'):
    for image_index, boxes in enumerate(box_list):
        if image_index >= 16:
            break
        image = image_array[image_index, :]
        for box in boxes:
            image = cv2.rectangle(np.transpose(image, [1, 2, 0]), (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), \
                color=[255, 0, 0], thickness=2)
            pass
        writer.add_image('{}/{}'.format(tag, image_index), np.transpose(image, [2, 0, 1]), global_step)
        pass
    pass