#In[]
# coding=utf-8
import numpy as np
import Putil.base.logger as plog

plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
plog.PutilLogConfig.config_format(plog.Format)


logger = plog.PutilLogConfig('TestCOCODataUnit').logger()
logger.setLevel(plog.DEBUG)

import Putil.data.coco as coco
from Putil.data.io_convertor import IOConvertorNoOp

coco.COCOData.set_seed(64)
coco_data = coco.COCOData('/data2/Public_Data/COCO/unzip_data/2017', coco.COCOData.Stage.Evaluate, './test/data/result/test_coco_data_unit', detection=True)
convertor = IOConvertorNoOp()
coco_data.set_convert_to_input_method(convertor)
data = coco_data[1]

#In[]:
import matplotlib.pyplot as plt
import numpy as np
import cv2

print(data[0].shape)
print(data[1].shape)
plt.imshow(data[0])
plt.show()
plt.imshow((data[1][:, :, 6] * 255).astype(np.uint8), cmap=plt.cm.gray)
print(cv2.resize(data[1][:, :, 6], (100, 200)).shape)
plt.show()