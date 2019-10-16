# coding=utf-8
import os
import json
import numpy as np
import Putil.base.logger as plog
import Putil.data.common_data as pcd

logger = plog.PutilLogConfig('coco').logger()
logger.setLevel(plog.DEBUG)
COCODataLogger = logger.getChild('COCOData')
COCODataLogger.setLevel(plog.DEBUG)


class COCOData(pcd.CommonData):
    def __init__(self, statistic_file, information_save_to='', load_truncate=None, k_fold=[0.9, 0.1], subset_class_amount=1000, data_drop_rate=0.0, **kwargs):
        '''
        focus on the coco2017:
        the data is for Segmentation, object detection, personal key-point detection and pic to word
        the semantic label and bbox label is in the instances_*, and the label is not completed
        the key point label is in the person_keypoints_*
        the captions label is in the captions_*
        '''
        pass

    def _restart_process(self, restart_param):
        pass

    def _inject_operation(self, inject_param):
        pass

    def _generate_from_specified(self, index):
        '''
        this function is call in the generate_data, using the specified id from the data_set_field to get the data from the id
        '''
        pass

    @staticmethod
    def statistic(coco_root='', year=''):
        '''
        generate a better statistic file for coco data, which should be easier to use
        '''
        train_root = os.path.join(coco_root, 'train{0}'.format(year))
        test_root = os.path.join(coco_root, 'test{0}'.format(year))
        val_root = os.path.join(coco_root, 'val{0}'.format(year))
        # get the label field, which data is unlabeled, which is labeled
        with open('/data/Public_Data/COCO/annotations/instances_train2017.json', 'r') as fp:
            instances = json.load(fp)
            pass
        # if the image does not exist, download the image
        # instances
        pass
    pass
