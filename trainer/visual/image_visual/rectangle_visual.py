# ccoding=utf-8
#In[]:
import re
import cv2
import numpy as np
from enum import Enum


class RectangleVisual:
    class BorderCrossType(Enum):
        Clip = 0
    def __init__(self, thickness, border_cross_type=BorderCrossType.Clip):
        self._thickness = thickness
        self._border_cross_type = border_cross_type
        pass

    def rectangle_visual(self, image, rectangles, classes=None, color_map=None, scores=None, **kwargs):
        '''
         @brief 
         @note
         @param[in] image
         shape [height, width, channel], type: uint8
         @param[in] rectangles 
         ndarray or list with shape [rectangle_amount, 4], type: float, format: [[top_left_x, top_left_y, width, height], ...]
         @param[in] classes default None
         ndarray or list with shape [rectangle_amount], type: int64
         @param[in] color_map
         list or map
        '''
        if len(image.shape) == 2:
            image = np.stack([image] * 3, -1)
            pass
        rectangles = np.array(rectangles)
        rectangles[:, 2: 4] = rectangles[:, 0: 2] + rectangles[:, 2: 4]
        if rectangles.dtype != np.int64:
            rectangles = rectangles.astype(np.int64)
            pass
        if self._border_cross_type == RectangleVisual.BorderCrossType.Clip:
            rectangles[:, [0, 2]] = np.clip(rectangles[:, [0, 2]], 0, image.shape[1])
            rectangles[:, [1, 3]] = np.clip(rectangles[:, [1, 3]], 0, image.shape[0])
        if classes is None:
            classes = [0] * len(rectangles)
        scores = [None] * len(rectangles) if scores is None else scores
        for rectangle, cla, sco in zip(rectangles, classes, scores):
            color = color_map[cla] if color_map is not None else [0, 255, 0]
            image = cv2.rectangle(image, pt1=tuple(rectangle[0: 2]), pt2=tuple(rectangle[2: 4]), color=color, thickness=self._thickness)
            #import pdb; pdb.set_trace()
            image = cv2.putText(image, str(sco).split('.')[0] + '.' + str(sco).split('.')[1][:3], \
                tuple(rectangle[0: 2]), fontFace=kwargs.get('fontFace', cv2.FONT_HERSHEY_SIMPLEX), fontScale=kwargs.get('fontScale', 0.9), color=tuple(color)) if sco is not None else image
        return image


#from PIL import Image
#import numpy as np
#import Putil.sampling.MCMC.metropolis_hasting as pmh
#from Putil.function.gaussian import Gaussian
#import matplotlib.pyplot as plt
#
#image = np.array(Image.open('/data2/process_data/caojihua/workspace/Putil/trainer/visual/image_visual/000000123413.jpg'))
#print(image.shape)
#
#rv = RectangleVisual(4)
#i = rv.rectangle_visual(image, [[10, 10, 250, 250], [150, -150, 200.0, 200]], [0, 1], {0: [0, 255, 0], 1: [255, 0, 0]})
#plt.imshow(i)
#plt.show()
#i = rv.rectangle_visual(image, [[0, 0, 300, 300], [50, 50, 200.0, 200]], color_map={0: [0, 255, 0], 1: [255, 0, 0]})
#plt.imshow(i)
#plt.show()
#i = rv.rectangle_visual(image, [[100, 100, 150, 150], [-50, 50, 500.0, 500]], scores=[0.9])
#plt.imshow(i)
#plt.show()