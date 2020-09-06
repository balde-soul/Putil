# coding=utf-8
import numpy as np
import cv2


class PointVisual:
    def __init__(self):
        pass

    def visual_on_image(self, image, point_weight, color_map):
        '''
         @brief
         @note make point visualization on the image
         @param[in] image [height, width, channel], support one or three channel
         @param[in] point_weight [height, width, point_type_amount], [0, 1]
        '''
        if len(image.shape) == 2:

        pass