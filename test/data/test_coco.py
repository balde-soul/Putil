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
import Putil.data.aug as pAug
import Putil.data.aug as pAug
from Putil.data.coco import COCOCommonAugBase
from Putil.data.data_type_adapter import DataTypeAdapterNoOp as data_type_adapter
from Putil.data.convert_to_input import ConvertToInputNoOp as convert_to_input


seed = 64

#COCO.COCOData.set_seed(seed)
#dataset_train = COCO.COCOData('/data2/Public_Data/COCO', COCO.COCOData.Stage.STAGE_TRAIN, './result', detection=True)
#root_node = pAug.AugNode(pAug.AugFuncNoOp())
#root_node.freeze_node()
#dataset_train.set_aug_node_root(root_node)
#class_amount = 80
#dataset_train.set_data_type_adapter(data_type_adapter())
#dataset_train.set_convert_to_input_method(convert_to_input())
#TestCocoLogger.info('train data amount: {0}'.format(len(dataset_train)))
#dataset_train[0]
#
#dataset_evaluate = COCO.COCOData('/data2/Public_Data/COCO', COCO.COCOData.Stage.STAGE_EVAL, './result', detection=True)
#root_node = pAug.AugNode(pAug.AugFuncNoOp())
#root_node.freeze_node()
#dataset_evaluate.set_aug_node_root(root_node)
#class_amount = 80
#dataset_evaluate.set_data_type_adapter(data_type_adapter())
#dataset_evaluate.set_convert_to_input_method(convert_to_input())
#TestCocoLogger.info('evaluate data amount: {0}'.format(len(dataset_evaluate)))

dataset_test = COCO.COCOData('/data2/Public_Data/COCO', COCO.COCOData.Stage.STAGE_TEST, './result', detection=True)
root_node = pAug.AugNode(pAug.AugFuncNoOp())
root_node.freeze_node()
dataset_test.set_aug_node_root(root_node)
class_amount = 80
dataset_test.set_data_type_adapter(data_type_adapter())
dataset_test.set_convert_to_input_method(convert_to_input())
TestCocoLogger.info('test data amount: {0}'.format(len(dataset_test)))
t = dataset_test[0]
print(t)