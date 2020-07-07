# coding=utf-8
#In[]:
import Putil.data.vision_common_convert.bbox_convertor as bbc
import numpy as np
import matplotlib.pyplot as plt


BC = bbc.BBoxConvertToCenterBox(sample_rate=4)
image = np.reshape(np.random.sample(10000), [100, 100])
print(image.shape)
image, label, weight = BC(image, [[60, 50, 30, 80]])
print(weight.shape)
plt.imshow((weight * 255).astype(np.uint8), cmap=plt.cm.gray)
plt.show()
#plt.imshow((label[:, :, 6] * 255).astype(np.uint8))
#plt.show()