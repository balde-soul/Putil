#In[]
# coding=utf-8

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Putil.data.vision_data_aug.image_aug import Contrast
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np

image_bgr = cv2.imread('./test/data/vision_data_aug/test_image.jpg')
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
plt.imshow(image_rgb)
plt.show()

image_hsl = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS)

image_hsl_full = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HLS_FULL)

#In[]
row = 70
col = 600
print(image_rgb[row, col, :])
print(image_hsl[row, col, :])
print(image_hsl_full[row, col, :])

#In[]
print(cv2.__version__)