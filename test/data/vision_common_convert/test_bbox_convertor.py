# coding=utf-8
#In[]:
import Putil.data.vision_common_convert.bbox_convertor as bbc
import numpy as np
import matplotlib.pyplot as plt


#t = np.reshape(np.random.sample(10), [2, 5])
#tt = np.reshape(np.random.sample(10), [2, 5])
#print(t)
#print(tt)
#print(np.max(np.stack([t, tt], axis=-1), axis=-1))
BC = bbc.BBoxConvertToCenterBox(sample_rate=4)
image = np.reshape(np.random.sample(10000), [100, 100])
print(image.shape)
image, label = BC(image, [[63.9, 51.9, 30.9, 80.9], [43.9, 33.9, 83.9, 63.9]])
print(label.shape)
weight = label[:, :, 6]
plt.imshow((weight * 255).astype(np.uint8), cmap=plt.cm.gray)
plt.show()
plt.imshow((label[:, :, 6] * 255).astype(np.uint8), cmap=plt.cm.gray)
plt.show()

##In[]
#import torch
#import numpy as np
#
#pool = torch.nn.AvgPool2d((2, 2), (2, 2))
#
#model = torch.nn.Sequential(
#    pool
#)
#
#x = torch.tensor(np.reshape(np.random.sample(49), [1, 1, 7, 7]))
#
#print(model(x))
#