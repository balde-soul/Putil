# coding=utf-8

import torch
import cv2


def torch_rectangle_image_summary(writer, image_tensor, batch_dim, dataformats='CHW'):
    image_amount = image_tensor.shape[batch_dim]
    image_c = torch.split(image_tensor, 1, batch_dim)
    for image_index in range(0, image_amount):
        image = image_c[image_index].detach().numpy()
        cv2.rectangle()
        pass
    pass