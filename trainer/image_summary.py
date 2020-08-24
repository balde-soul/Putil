# coding=utf-8

import numpy as np
import torch
import cv2


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