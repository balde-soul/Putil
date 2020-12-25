#In[]:
# coding=utf-8
import os
#import pdb; pdb.set_trace()
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import Putil.base.logger as plog

plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(plog.FormatRecommend)
plog.PutilLogConfig.config_log_level(stream=plog.DEBUG)
root_logger = plog.PutilLogConfig('test_coco').logger()
root_logger.setLevel(plog.DEBUG)
TestCocoLogger = root_logger.getChild('TestCoco')
TestCocoLogger.setLevel(plog.DEBUG)

import Putil.data.coco as COCO
from importlib import reload
reload(COCO)
import Putil.data.aug as pAug
import Putil.data.aug as pAug
from Putil.data.data_type_adapter import DataTypeAdapterNoOp as data_type_adapter
from Putil.data.convert_to_input import ConvertToInputNoOp as convert_to_input


seed = 64
image_height = 512
image_width = 512
root_dir = '/data2/Public_Data/COCO/unzip_data/2017'
information_save_to_path = './test/data/result/test_coco'
if os.path.exists(information_save_to_path) is False:
    os.mkdir(information_save_to_path)
#In[]
dataset_test = COCO.COCOData(root_dir, COCO.COCOData.Stage.Test, information_save_to_path, detection=True, 
image_height=image_height, image_width=image_width)
root_node = pAug.AugNode(pAug.AugFuncNoOp())
root_node.freeze_node()
dataset_test.set_aug_node_root(root_node)
class_amount = 80
dataset_test.set_data_type_adapter(data_type_adapter())
dataset_test.set_convert_to_input_method(convert_to_input())
TestCocoLogger.info('test data amount: {0}'.format(len(dataset_test)))
t = dataset_test[0]
#In[]
dataset_evaluate = COCO.COCOData(root_dir, COCO.COCOData.Stage.Evaluate, information_save_to_path, detection=True, 
image_height=image_height, image_width=image_width)
root_node = pAug.AugNode(pAug.AugFuncNoOp())
root_node.freeze_node()
dataset_evaluate.set_aug_node_root(root_node)
class_amount = 80
dataset_evaluate.set_data_type_adapter(data_type_adapter())
dataset_evaluate.set_convert_to_input_method(convert_to_input())
TestCocoLogger.info('evaluate data amount: {0}'.format(len(dataset_evaluate)))
#In[]
dataset_evaluate = COCO.COCOData(root_dir, COCO.COCOData.Stage.Evaluate, information_save_to_path=information_save_to_path, detection=True, 
image_height=image_height, image_width=image_width, cat_ids=list(COCO.COCOBase.cat_id_to_cat_name.keys())[0: 2])
root_node = pAug.AugNode(pAug.AugFuncNoOp())
root_node.freeze_node()
dataset_evaluate.set_aug_node_root(root_node)
class_amount = 80
dataset_evaluate.set_data_type_adapter(data_type_adapter())
dataset_evaluate.set_convert_to_input_method(convert_to_input())
TestCocoLogger.info('evaluate data amount: {0}'.format(len(dataset_evaluate)))
#In[]
COCO.COCOData.set_seed(seed)
dataset_train = COCO.COCOData(root_dir, COCO.COCOData.Stage.Train, information_save_to_path, detection=True, use_rate=0.1, 
image_height=image_height, image_width=image_width)
root_node = pAug.AugNode(pAug.AugFuncNoOp())
root_node.freeze_node()
dataset_train.set_aug_node_root(root_node)
class_amount = 80
dataset_train.set_data_type_adapter(data_type_adapter())
dataset_train.set_convert_to_input_method(convert_to_input())
TestCocoLogger.info('train data amount: {0}'.format(len(dataset_train)))
t = dataset_train[0]
#In[]:
dataset = dataset_evaluate
import matplotlib.pyplot as plt
import cv2
ret = dataset_evaluate[100]
print(len(ret))
image = ret[0]
boxes = ret[1]
base_information = ret[2]
print('base_information{}'.format(base_information))
classes = ret[-1]
plt.imshow(ret[0])
plt.show()
for box in boxes:
    center_x = int(box[0] + 0.5 * box[2])
    center_y = int(box[1] + 0.5 * box[3])
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[0] + box[2])
    y2 = int(box[1] + box[3])
    image = cv2.circle(image, (center_x, center_y), radius=1, thickness=2, color=[255, 0, 0])
    image = cv2.rectangle(image, (x1, y1), (x2, y2), thickness=2, color=[0, 255, 0])
plt.imshow(image)
plt.show()