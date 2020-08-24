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
BC = bbc.BBoxConvertToCenterBox(sample_rate=4, class_amount=10, \
    io=bbc.BBoxConvertToCenterBox.IODirection.InputConvertion)
image = np.reshape(np.random.sample(10000), [100, 100])
print(image.shape)
image, box_label, class_label, obj_label, radiance_factor  = BC(image, [[63.9, 51.9, 30.9, 80.9], [43.9, 33.9, 83.9, 63.9]], [1, 2])
print(box_label.shape)
plt.imshow((radiance_factor * 255).astype(np.uint8), cmap=plt.cm.gray)
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

#In[]:
import Putil.data.vision_common_convert.bbox_convertor as bbc
import numpy as np
import torch
import matplotlib.pyplot as plt

BC = bbc.BBoxConvertToCenterBox(sample_rate=4, class_amount=10, \
    io=bbc.BBoxConvertToCenterBox.IODirection.OutputConvertion)
images = np.reshape(np.random.sample(1440000), [3, 3, 400, 400])

class_out = np.reshape(np.random.sample(90000), [3, 3, 100, 100])
obj_out = np.reshape(np.random.sample(30000), [3, 1, 100, 100])
box_out = np.reshape(np.random.sample(120000) * 100, [3, 4, 100, 100])
ret = BC(images, box_out, class_out, obj_out)
