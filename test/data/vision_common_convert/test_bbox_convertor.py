# coding=utf-8
import Putil.data.vision_common_convert.bbox_convertor as bbc
import numpy as np


BC = bbc.BBoxConvertToCenterBox(sample_rate=4)
image = np.reshape(np.random.sample(10000), [100, 100])
print(image.shape)
BC(image, [[60, 50, 30, 50]])