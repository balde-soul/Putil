#In[]:
# coding=utf-8
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
from importlib import reload;reload(COCO)
from importlib import reload
reload(COCO)
import Putil.data.aug as pAug
import Putil.data.aug as pAug
from Putil.data.coco import COCOCommonAugBase
from Putil.data.data_type_adapter import DataTypeAdapterNoOp as data_type_adapter
from Putil.data.convert_to_input import ConvertToInputNoOp as convert_to_input


seed = 64
image_height = 512
image_width = 512
root_dir = '/data2/Public_Data/COCO/unzip_data/2017'

#In[]
dataset_evaluate = COCO.COCOData(root_dir, COCO.COCOData.Stage.STAGE_EVAL, './test/data/result', detection=True, 
image_height=image_height, image_width=image_width)
root_node = pAug.AugNode(pAug.AugFuncNoOp())
root_node.freeze_node()
dataset_evaluate.set_aug_node_root(root_node)
class_amount = 80
dataset_evaluate.set_data_type_adapter(data_type_adapter())
dataset_evaluate.set_convert_to_input_method(convert_to_input())
TestCocoLogger.info('evaluate data amount: {0}'.format(len(dataset_evaluate)))

#In[]:
print(dataset_evaluate[0])
def get_detection_result(*args):
    image = args[0]
    bboxes = args[1]
    base_informations = args[2]
    category_ids = args[-1]
    return image, base_informations[-1], category_ids, bboxes, [1.0 for bbox in bboxes]
    pass
use_amount = 300
for i in range(0, use_amount):
    dataset_evaluate.add_detection_result(*get_detection_result(*dataset_evaluate[i]), save=True if i == use_amount - 1 else False)
    pass
#for i in range(10, len(dataset_evaluate)):
#    dataset_evaluate.add_detection_result(*get_detection_result(*dataset_evaluate[i]), save=True if i == len(dataset_evaluate) else False)
#    pass
#In[]
#print(dataset_evaluate._detection_result)
#len(dataset_evaluate)
dataset_evaluate.evaluate_detection()