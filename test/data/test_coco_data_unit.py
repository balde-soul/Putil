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
from Putil.data.vision_common_convert.bbox_convertor import BBoxConvertToCenterBox as Convertor 

coco.COCOData.set_seed(64)
coco_data = coco.COCOData('/data2/Public_Data/COCO/unzip_data/2017', coco.COCOData.Stage.STAGE_EVAL, '', detection=True)
convertor = Convertor(4, sigma=np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32))
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