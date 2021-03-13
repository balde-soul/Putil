#In[]
# coding=utf-8
from Putil.base.jupyter_state import go_to_top
import os
go_to_top(4, os.path.abspath(__file__))

import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np
from Putil.data.vision_data_aug.image_aug import Saturation
from Putil.test.data.vision_data_aug._test_base import contrast

image = cv2.imread('./test/data/vision_data_aug/test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
contrast(image)
plt.imshow(image)
plt.show()
#image = np.zeros(shape=[100, 100, 3], dtype=np.float32)
saturation = Saturation()
saturation.increment = 1.0
image_saturation = saturation(image)

plt.imshow(image_saturation)
plt.show()