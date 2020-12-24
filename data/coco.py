# coding=utf-8
from pycocotools.cocoeval import COCOeval
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
from pandas.plotting import table
from collections import OrderedDict
import matplotlib.pyplot as plt
import pandas as pd
from collections import Iterable
import re
import pandas as pd
from abc import ABCMeta, abstractmethod
import random
import json
from skimage import io
#import matplotlib.pyplot as plt
import cv2
import time
from enum import Enum
from PIL import Image
import os
import json
import numpy as np
import Putil.base.logger as plog
from pycocotools.coco import COCO

logger = plog.PutilLogConfig('coco').logger()
logger.setLevel(plog.DEBUG)
COCODataLogger = logger.getChild('COCOData')
COCODataLogger.setLevel(plog.DEBUG)
COCOBaseLogger = logger.getChild('COCOBase')
COCOBaseLogger.setLevel(plog.DEBUG)

import Putil.data.vision_common_convert.bbox_convertor as bbox_convertor
from Putil.data.util.vision_util.detection_util import rect_angle_over_border as rect_angle_over_border
from Putil.data.util.vision_util.detection_util import clip_box_using_image as clip_box 
import Putil.data.common_data as pcd


class COCOBase():
    '''
     @brief
     @note
      有关coco的信息，总共有四类任务：目标检测detection、语意分割stuff、全景分割pann
    '''
    # represent->cat_id->cat_name->represent
    _detection_represent_to_cat_id = OrderedDict({0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90})
    _detection_cat_id_to_represent = OrderedDict()
    for represent, cat_id in _detection_represent_to_cat_id.items():
        _detection_cat_id_to_represent[cat_id] = represent
    #_detection_cat_id_to_represent = {cat_id: represent for represent, cat_id in _detection_represent_to_cat_id.items()}
    _detection_cat_id_to_cat_name = OrderedDict({1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'})
    _detection_cat_name_to_represent = OrderedDict()
    for cat_id, cat_name in _detection_cat_id_to_cat_name.items():
        _detection_cat_name_to_represent[cat_name] = _detection_cat_id_to_represent[cat_id]
    # TODO: important problem remain 当使用以下方法生成_detection_cat_name_to_represent时出现_detection_cat_id_to_represent undefined的情况
    #_detection_cat_name_to_represent = {cat_name: _detection_cat_id_to_represent[cat_id] for cat_id, cat_name in _detection_cat_id_to_cat_name.items()}
    # datas field
    ## base information
    base_information_length = 3
    image_height_index_in_base_information = 0
    image_width_index_in_base_information = 1
    image_id_index_in_base_information = 2
    ## datas
    data_length = 4
    image_index = 0
    detection_box_index = 1
    base_information_index = 2
    detection_class_index = 3
    ## result
    result_length = 5 # format: list
    result_base_information_index = 0
    result_image_index = 1
    result_detection_box_index = 2 # format: [[top_x, top_y, width, height], ...]
    result_detection_class_index = 3 # format: [class_represent] class_represent表示的是当前class使用的索引号，不是cat_id
    result_detection_class_score = 4

    @staticmethod
    def generate_base_information(image_ann):
        base_information = [None] * COCOBase.base_information_length
        base_information[COCOBase.image_height_index_in_base_information] = image_ann[0]['height']
        base_information[COCOBase.image_width_index_in_base_information] = image_ann[0]['width']
        base_information[COCOBase.image_id_index_in_base_information] = image_ann[0]['id']
        return base_information

    @staticmethod
    def generate_default_datas():
        return [None] * COCOBase.data_length

    @staticmethod
    def generate_default_result():
        return [None] * COCOBase.result_length

    @staticmethod
    def detection_get_cat_id(cat_name=None, represent_value=None):
        assert False in [t is None for t in [cat_name, represent_value]]
        return COCOBase._detection_represent_to_cat_id[COCOBase._detection_cat_name_to_represent[cat_name]] if cat_name is not None else COCOBase._detection_represent_to_cat_id[represent_value]

    @staticmethod
    def detection_get_cat_name(cat_id=None, represent_value=None):
        assert False in [t is None for t in [cat_id, represent_value]]
        return COCOBase._detection_cat_id_to_cat_name[cat_id] if cat_id is not None else COCOBase._detection_cat_id_to_cat_name[COCOBase._detection_represent_to_cat_id[represent_value]]

    @staticmethod
    def detection_statistic_obj_size_follow_cat(cat_names, ann_file, save_to):
        cat_ids = [COCOBase._detection_represent_to_cat_id[COCOBase._detection_cat_name_to_represent[cat_name]] for cat_name in cat_names] if type(cat_names).__name__ == 'list'\
            else [COCOBase._detection_represent_to_cat_id[COCOBase._detection_cat_name_to_represent[cat_names]]]
        coco = COCO(ann_file)
        #row_amount = np.floor(np.sqrt(len(cat_ids)))
        #col_amount = row_amount
        #plt.figure(figsize=(row_amount, col_amount))
        #fig = plt.figure()
        #fig.suptitle('y: counts, x: bbox area/1000')
        for index, cat_id in enumerate(cat_ids):
            #plt.subplot(row_amount, col_amount, index + 1)
            ann_ids = coco.getAnnIds(catIds=[cat_id])
            anns = coco.loadAnns(ann_ids)
            anns_df = pd.DataFrame(anns)
            bbox_area = anns_df['bbox'].apply(lambda x: x[2] * x[3])
            plt.rcParams['savefig.dpi'] = 300
            (bbox_area/100).plot.hist(grid=True, bins=500, rwidth=0.9, color='#607c8e')
            plt.title(COCOBase._detection_cat_id_to_cat_name[cat_id])
            plt.ylabel('Counts')
            plt.xlabel('bbox area/100')
            plt.savefig(os.path.join(save_to, 'box_area_histogram_{}.png'.format(COCOBase._detection_cat_id_to_cat_name[cat_id])))
            plt.close()
            #plt.show()
            #hist, xedges, yedges = np.histogram2d(anns_df['bbox'].apply(lambda x: x[2]), anns_df['bbox'].apply(lambda x: x[3]), bins=1000)
            pass
        pass

    @staticmethod
    def detection_statistic_img_amount_obj_amount(ann_file, save_to, cat_name=None):
        coco = COCO(ann_file)
        if cat_name is not None:
            cat_ids = [COCOBase._detection_represent_to_cat_id[COCOBase._detection_cat_name_to_represent[cat_name]] for cat_name in cat_names] if type(cat_names).__name__ == 'list'\
                else [COCOBase._detection_represent_to_cat_id[COCOBase._detection_cat_name_to_represent[cat_names]]]
            pass
        else:
            cat_ids = coco.getCatIds()
            pass
        result = list()
        for cat_id in cat_ids:
            img_id = coco.getImgIds(catIds=[cat_id])
            ann_id = coco.getAnnIds(catIds=[cat_id])
            result.append({'category': COCOBase._detection_cat_id_to_cat_name[cat_id], 'img_amount': len(img_id), \
                'cat_id': cat_id, 'obj_amount': len(ann_id)})
            pass
        result_df = pd.DataFrame(result)
        plt.rcParams['savefig.dpi'] = 300
        fig = plt.figure(figsize=(5, 15))
        ax = fig.add_subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        table(ax, result_df, loc='center')
        plt.savefig(os.path.join(save_to, 'category_img_amount.png'))
        pass

    @staticmethod
    def detection_statistic_obj_size_follow_img(img_id, ann_file):
        pass

    def __init__(
        self,
        coco_root_dir,
        stage,
        information_save_to_path,
        detection,
        key_points,
        stuff,
        panoptic,
        dense_pose,
        captions,
        cat_ids,
    ):
        self._information_save_to_path = information_save_to_path
        self._coco_root_dir = coco_root_dir
        self._stage = stage
        self._img_root_name = 'train2017' if self._stage == COCOData.Stage.Train else \
            ('val2017' if self._stage == COCOData.Stage.Evaluate else 'test2017')
        self._img_root_dir = os.path.join(self._coco_root_dir, self._img_root_name)
        self._detection = detection
        self._key_points = key_points
        self._stuff = stuff
        self._panoptic = panoptic
        self._dense_pose = dense_pose
        self._captions = captions
        assert True in [self._detection, self._key_points, self._stuff, self._panoptic, self._dense_pose, self._captions]

        self._cat_ids = cat_ids
        COCOBaseLogger.info('specified cat_ids: {}'.format(self._cat_ids)) if self._cat_ids is not None else None
        
        self._instances_file_train = os.path.join(self._coco_root_dir, 'annotations/instances_train2017.json')
        self._instances_file_eval = os.path.join(self._coco_root_dir, 'annotations/instances_val2017.json')
        self._person_keypoints_train = os.path.join(self._coco_root_dir, 'annotations/person_keypoints_train2017.json')
        self._person_keypoints_eval = os.path.join(self._coco_root_dir, 'annotations/person_keypoints_val2017.json')
        self._captions_train = os.path.join(self._coco_root_dir, 'annotations/captions_train2017.json')
        self._captions_eval = os.path.join(self._coco_root_dir, 'annotations/captions_val2017.json')
        self._image_info_test = os.path.join(self._coco_root_dir, 'annotations/image_info_test2017.json')

        # result
        self._detection_result = None
        self._detection_result_file_name = 'detection_result'
        # detection indicator result
        self._detection_indicator_result = None
        self._detection_indicator_result_file_name = 'detection_indicator_result'
        pass

    def read_image(self, file_name):
        image = cv2.imread(os.path.join(self._img_root_dir, file_name)).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        assert(image is not None)
        return image

    def represent_value_to_category_id(self, represent_value):
        pass

    def add_result(self, result, save=False, prefix=None):
        if self._detection:
            #self.add_detection_result(
            #    image=result[COCOBase.result_image_index], 
            #    image_id=result[COCOBase.result_base_information_index][COCOBase.image_id_index_in_base_information],
            #    category_ids=)
            raise NotImplementedError('save the detection result is not implemented')
            pass
        elif self._stuff:
            raise NotImplementedError('save the detection result is not implemented')
            pass
        else:
            raise NotImplementedError('save the detection result is not implemented')
            pass
        pass

    @staticmethod
    def generate_result_file_name(prefix, common_name):
        return '{}{}.csv'.format('{}_'.format(prefix) if prefix is not None else '', common_name)

    def add_detection_result(self, image=None, image_id=None, category_ids=None, bboxes=None, scores=None, save=False, prefix=None):
        '''
         @brief save the detection result base on one image
         @note
         @param[in] image ndarray the image
         @param[in] image_id int the id of the image 
         @param[in] category_ids list|[int|category_id, ...] the category of the bboxes
         @param[in] bboxes list|[list|[float|top_left_x, float|top_left_y, float|width, float|height], ...]
         @param[in] scores list|[float|score, ...]
         @param[in] save 
         bool 
         save the result to the file or not, if True the _detection_result would be saved to _detection_result file,
         _detection_result would be set as None, _detection_result_file would be changed
         @param[in] prefix str the prefix of the file name to save the result
        '''
        sync_status = [list(image), image_id, list(category_ids), list(bboxes), list(scores)]
        if None in sync_status:
            assert(len(set(sync_status)) == 1), COCODataLogger.fatal('all element should be None while None in sync_status: {}'.format(sync_status))
            pass
        else:
            used_wh = image.shape[0: 2][::-1]
            self._detection_result = pd.DataFrame() if self._detection_result is None else self._detection_result
            result_temp = list()
            for category_id, bbox, score in zip(category_ids, bboxes, scores):
                result_temp.append({'image_id': image_id, 'category_id': category_id, 'bbox': bbox, 'score': score})
            self._detection_result = self._detection_result.append(result_temp, ignore_index=True)
            pass
        if save:
            if self._detection_result is not None:
                # : save the _detection_result
                self._detection_result.set_index(['image_id'], inplace=True)
                detection_result_file_path = os.path.join(self._information_save_to_path, \
                    COCOBase.generate_result_file_name(prefix, self._detection_result_file_name))
                self._detection_result.to_csv(detection_result_file_path)
                pass
            self._detection_result = None
            #self._detection_result_file_name = \
            #'{}_{}-{}.csv'.format(prefix, self._detection_result_file_name.split('.')[0], \
            #    1 if len(self._detection_result_file_name.split('.')[0].split('-')) == 1 else int(self._detection_result_file_name.split('.')[0].split('-')[-1]) + 1)
            #self._detection_result_file = os.path.join(self._information_save_to_path, self._detection_result_file_name)
            pass
        pass

    def evaluate_detection(self, image_ids=None, cat_ids=None, prefix=None):
        '''
         @brief evaluate the performance
         @note use the result files in the self._information_save_to_path, combine all result files and save to a json file, and
         then we would use this json file to evaluate the performance, base on object the image_ids Cap cat_ids
         @param[in] image_ids the images would be considered in the evaluate
         @param[in] cat_ids the categories would be considered in the evaluate
        '''
        assert type(prefix).__name__ == 'list' or prefix is None or type(prefix).__name__ == 'str'
        target_files = [COCOBase.generate_result_file_name(prefix, self._detection_result_file_name) for _prefix in prefix] if type(prefix).__name__ == 'list' \
            else [COCOBase.generate_result_file_name(prefix, self._detection_result_file_name)]
        detection_result = None
        for target_file in target_files:
            detection_result_temp = pd.read_csv(os.path.join(self._information_save_to_path, target_file), \
                converters={'bbox': lambda x: [float(t.strip('[').strip(']')) for t in x.split(',')]})
            if detection_result is not None:
                detection_result = detection_result.append(detection_result_temp)
            else:
                detection_result = detection_result_temp
                pass
            pass
        index_name = {index: name for index, name in enumerate(list(detection_result.columns))}
        detection_result_formated = [{index_name[index]: tt for index, tt in enumerate(t)} for t in list(np.array(detection_result))]

        with open(os.path.join(self._information_save_to_path, 'formated_detection_result.json'), 'w') as fp:
            json.dump(detection_result_formated, fp)
        
        detection_result_coco = self._instances_coco.loadRes(os.path.join(self._information_save_to_path, 'formated_detection_result.json'))
        #result_image_ids = detection_result_coco.getImgIds()
        cocoEval = COCOeval(self._instances_coco, detection_result_coco, 'bbox')
        cocoEval.params.imgIds  = image_ids if image_ids is not None else self._instances_coco.getImgIds()
        cocoEval.params.catIds = cat_ids if cat_ids is not None else self._instances_coco.getCatIds()
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        pass

    def evaluate_segmentation(self, image_ids, cat_ids, prefix):
        raise NotImplementedError('evaluate segmentation is not implemented')
        pass


    def evaluate_keypoint(self, image_ids, cat_ids, prefix):
        raise NotImplementedError('evaluate keypoint is not implemented')
        pass
    

    def evaluate_panoptic(self, image_ids, cat_ids, prefix):
        raise NotImplementedError('evaluate panoptic is not implemented')
        pass


    def get_detection_indicator(self, scores, ious, pre_file, image_ids=None, cat_ids=None):
        '''
         @brief
         @note calculate the map
        '''
        target_image_ids = image_ids if image_ids is not None else self._data_field
        target_cat_ids = cat_ids if cat_ids is not None else self._cat_ids
        pass
    pass


class COCOData(pcd.CommonDataForTrainEvalTest, COCOBase):
    def save_result(self, *result):
        #if self._detection and not self._stuff and not self._p
        pass

    @staticmethod
    def set_seed(seed):
        pcd.CommonDataWithAug.set_seed(seed)
        pass

    def set_detection_label_generator(self, generator):
        pass

    def set_key_points_label_generator(self, generator):
        pass

    def set_stuff_label_generator(self, generator):
        pass

    def set_panoptic_label_generator(self, generator):
        pass

    def set_dense_pose_label_generator(self, generator):
        pass

    def set_captions_label_generator(self, generator):
        pass
    
    @staticmethod
    def __get_common_id(id_lists):
        if len(id_lists) > 1:
            common_list = list()
            for sample in id_lists[0]:
                view = [(sample in id_list) if id_list is not None else True for id_list in id_lists[1:]]
                common_list.append(sample) if False not in view else None
                pass
            return common_list
        else:
            return id_lists[0]
        pass

    def __init__(
        self, 
        coco_root_dir, 
        stage,
        information_save_to_path=None,
        detection=False,
        key_points=False,
        stuff=False,
        panoptic=False,
        dense_pose=False,
        captions=False,
        cat_ids=None,
        use_rate=1.0,
        image_width=128,
        image_height=128,
        remain_strategy=pcd.CommonData.RemainStrategy.Drop):
        '''
         @brief focus on coco2017
         @note 
         @param[in] stage 
         the stage of the dataset, Stage.STAGE_TRAIN,Stage.STAGE_EVAL or Stage.STAGE_TEST
         @param[in] coco_root_dir
         the root dir of the coco, the annotations is the path which contain the ann files
         @param[in] information_save_to_path
         the path to save the data information
         @param[in] detection
         read the detection label or not(in the file: instances)
         @param[in] stuff
         read the stuff label or not(in the file instances)
         @param[in] panoptic
         read the panoptic label or not(in the file)
         @param[in] dense_pose
         read the dense_pose label or not
         @param[in] captions
         read the captions label or not
         @param[in] cat_ids 
         used as sub_data
         @param[in] use_rate
         data used rate
        '''
        self._image_width = image_width
        self._image_height = image_height
        COCOBase.__init__(self, coco_root_dir, stage, information_save_to_path, detection, \
            key_points, stuff, panoptic, dense_pose, captions, cat_ids)
        pcd.CommonDataWithAug.__init__(self, use_rate=use_rate, sub_data=cat_ids, remain_strategy=remain_strategy)

        belong_instances = [self._detection, self._stuff, self._panoptic]
        belong_person_keypoints = [self._key_points]
        belong_captions = [self._captions]

        with_label = [COCOData.Stage.Train, COCOData.Stage.Evaluate]
        without_label = [COCOData.Stage.Test]
        self._instances_coco, instances_load = (COCO(self._instances_file_train \
            if self._stage == COCOData.Stage.Train else self._instances_file_eval), True) \
                if ((self._stage in with_label) and (True in [self._detection, self._stuff, self._panoptic])) else (None, False)
        self._instances_img_ids = self._instances_coco.getImgIds() if instances_load else list() 
        self._person_keypoints_coco, key_point_load = (COCO(self._person_keypoints_train \
            if self._stage == COCOData.Stage.Train else self._person_keypoints_eval), True) \
                if ((self._stage in with_label) and (self._key_points)) else (None, False)
        self._persion_keypoints_img_ids = self._person_keypoints_coco.getImgIds() if key_point_load else list()
        self._captions_coco, captions_load = (COCO(self._captions_train \
            if self._stage == COCOData.Stage.Train else self._captions_eval), True) \
                if ((self._stage in with_label) and (self._captions)) else (None, False)
        self._captions_img_ids = self._captions_coco.getImgIds() if captions_load else list()
        self._image_test, image_test_load = (COCO(self._image_info_test), True) if self._stage in without_label else (None, False)
        self._image_test_img_ids = self._image_test.getImgIds() if image_test_load else list()

        assert [instances_load, key_point_load, captions_load, image_test_load].count(True) == 1, "only support one ann file"

        # we know use the detectio only
        #self._data_field = COCOData.__get_common_id([self._instances_img_ids, self._persion_keypoints_img_ids, \
        #     self._captions_img_ids, self._image_test_img_ids])
        # TODO:record
        self._data_field = self._instances_img_ids + self._persion_keypoints_img_ids + self._captions_img_ids + self._image_test_img_ids
        if self._stage in [COCOData.Stage.Train, COCOData.Stage.Evaluate]:
            self._data_field = self._instances_coco.getImgIds(catIds=self._sub_data) if self._sub_data is not None else self._instances_img_ids 
            self._detection_cat_id_to_represent = COCOBase._detection_cat_id_to_represent if self._sub_data is None else {cat_id: index for index, cat_id in enumerate(self._sub_data)}
            if self._information_save_to_path is not None:
                with open(os.path.join(self._information_save_to_path, 'detection_cat_id_to_represent.json'), 'w') as fp:
                    json.dump(self._detection_cat_id_to_represent, fp, indent=4)
        self._fix_field()
     
        # check the ann
        if self._stage in with_label:
            image_without_ann = dict()
            for index in self._data_field:
                image_ann = self._instances_coco.loadImgs(index)
                ann_ids = self._instances_coco.getAnnIds(index)
                if len(ann_ids) == 0:
                    image_without_ann[index] = image_ann
            for index_out in list(image_without_ann.keys()):
                self._data_field.remove(index_out)
            with open('./image_without_ann.json', 'w') as fp:
                str_ = json.dumps(image_without_ann, indent=4)
                fp.write(str_)
                pass
        pass

    def _restart_process(self, restart_param):
        self._image_width = restart_param('image_width', self._image_width)
        self._image_height = restart_param.get('image_height', self._image_height)
        pass

    def _inject_operation(self, inject_param):
        pass

    def __generate_base_image_information(self, image_ann):
        #import pdb;pdb.set_trace()
        return [image_ann[0]['height'], image_ann[0]['width'], image_ann[0]['id']]

    class BaseInformationIndex(Enum):
        ImageHeightIndex = 0
        ImageWidthIndex = 1
        ImageIdIndex = -1

    @staticmethod
    def get_image_height(base_information):
        return base_information[COCOData.BaseInformationIndex.ImageHeightIndex]
    
    def get_image_width(base_information):
        return base_information[COCOData.BaseInformationIndex.ImageWidthIndex]

    def get_image_id(base_information):
        return base_information[COCOData.BaseInformationIndex.ImageIdIndex]

    def _generate_from_origin_index(self, index):
        '''
         @brief generate the image [detection_label ]
         @note
         @ret 
         [0] image [height, width, channel] np.float32
         [1] bboxes list float [[top_x, top_y, width, height], ....(boxes_amount)]
         [2] base_information list|[list|[int|image_height, int|image_width, int|image_id], ...]
         [-1] classes list int [class_index, ....] 0 for the background class may not equal with the category_id
        '''
        if self._stage == COCOData.Stage.Test:
            return self.__generate_test_from_origin_index(index)
        elif True in [self._detection, self._stuff, self._panoptic]:
            datas = COCOBase.generate_default_datas()
            image_ann = self._instances_coco.loadImgs(self._data_field[index])
            base_information = COCOBase.generate_base_information(image_ann)
            ann_ids = self._instances_coco.getAnnIds(self._data_field[index])
            anns = self._instances_coco.loadAnns(ann_ids)
            image = self.read_image(image_ann[0]['file_name'])
            # debug check
            for ann in anns:
                box = ann['bbox']
                if (box[0] + box[2] > image.shape[1]) or (box[1] + box[3] > image.shape[0]):
                    COCODataLogger.info(box)
                    pass
                pass
            #plt.axis('off')
            ##COCODataLogger.debug(image.shape)
            #plt.imshow(image)
            resize_width = self._image_width
            resize_height = self._image_height
            x_scale = float(resize_width) / image.shape[1]
            y_scale = float(resize_height) / image.shape[0]
            image = cv2.resize(image, (resize_width, resize_height), interpolation=Image.BILINEAR)
            #self._instances_coco.showAnns(anns, draw_bbox=True)
            #plt.show()
            bboxes = list()
            classes = list()
            for ann in anns:
                if self._cat_ids is None:
                    pass
                elif ann['category_id'] not in self._cat_ids:
                    continue
                box = ann['bbox']
                classes.append(self._detection_cat_id_to_represent[ann['category_id']])
                #bboxes.append([(box[0] + 0.5 * box[2]) * x_scale, (box[1] + 0.5 * box[3]) * y_scale, box[2] * x_scale, box[3] * y_scale])
                bboxes.append([box[0] * x_scale, box[1] * y_scale, box[2] * x_scale, box[3] * y_scale])
                pass
            #for box in bboxes:
            #    cv2.rectangle(image, (box[0] - box[])
            #assert rect_angle_over_border(bboxes, image.shape[1], image.shape[0]) is False, "cross the border"
            #if index == 823:
            #    pass
            bboxes = clip_box(bboxes, image)
            classes = np.delete(classes, np.argwhere(np.isnan(bboxes)), axis=0)
            bboxes = np.delete(bboxes, np.argwhere(np.isnan(bboxes)), axis=0)
            datas[COCOBase.base_information_index] = base_information
            datas[COCOBase.image_index] = image
            datas[COCOBase.detection_box_index] = bboxes
            datas[COCOBase.detection_class_index] = classes
            #ret = self._aug_check(*ret)
            COCODataLogger.warning('original data generate no obj, regenerate') if len(datas[COCOBase.detection_box_index]) == 0 else None
            return tuple(datas)
        else:
            raise NotImplementedError('unimplemented')
            pass
        pass

    def _aug_check(self, *args):
        if self._stage == COCOData.Stage.Train or (self._stage == COCOData.Stage.Evaluate):
            if True in [self._detection, self._stuff, self._panoptic]:
                bboxes = args[COCOBase.detection_box_index]
                classes = args[COCOBase.detection_class_index]
                assert len(bboxes) == len(classes)
                COCODataLogger.warning('zero obj occu') if len(bboxes) == 0 else None
                if len(bboxes) == 0:
                    pass
                assert np.argwhere(np.isnan(np.array(bboxes))).size == 0
                pass
            else:
                # TODO: other type
                pass
        elif self._stage == COCOData.Stage.Test:
            pass
        else:
            raise ValueError('stage: {} not supported'.format(self._stage))
            pass
        return args

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

    def __generate_test_from_origin_index(self, index):
        image_ann = self._image_test.loadImgs(self._image_test_img_ids[index])
        image = self.read_image(image_ann[0]['file_name'])
        resize_width = self._image_width
        resize_height = self._image_height
        x_scale = float(resize_width) / image.shape[1]
        y_scale = float(resize_height) / image.shape[0]
        image = cv2.resize(image, (resize_width, resize_height), interpolation=Image.BILINEAR)
        return image, image_ann[0]['id']

    def __generate_instance_from_origin_index(self, index):
        pass

    def __generate_keypoint_from_origin_index(self, index):
        pass

    def __generate_caption_from_origin_index(self, index):
        pass
    pass


pcd.CommonDataManager.register('COCOData', COCOData)


class SubCOCOData(COCOData):
    def __init__(self):
        pass


from torch.utils.data import Dataset


class COCODataWithTorch(COCOData, Dataset):
    def __init__(
        self, 
        coco_root_dir, 
        stage,
        information_save_to_path=None,
        detection=False,
        key_points=False,
        stuff=False,
        panoptic=False,
        dense_pose=False,
        captions=False,
        cat_ids=None,
        use_rate=1.0,
        image_width=128,
        image_height=128,
        remain_strategy=None):
        COCOData.__init__(self, coco_root_dir=coco_root_dir, stage=stage, information_save_to_path=information_save_to_path, \
            detection=detection, key_points=key_points, stuff=stuff, panoptic=panoptic, dense_pose=dense_pose, captions=captions, \
                cat_ids=cat_ids, use_rate=use_rate, image_height=image_height, image_width=image_width, remain_strategy=remain_strategy)
        Dataset.__init__(self)
        pass
    pass


#class COCOCommonAugBase:
#    instance_image_index = 0
#    image_index = instance_image_index
#    instance_bboxes_index = 1
#    bboxes_index = instance_bboxes_index
#    base_information_index = 2
#    instance_classes_index= -1
#    classes_index = instance_classes_index
#
#    @staticmethod
#    def _repack(*original_input, image=None, bboxes=None, classes=None):
#        #import pdb; pdb.set_trace()
#        image = image if image is not None else original_input[COCOCommonAugBase.image_index]
#        bboxes = np.array(bboxes if bboxes is not None else original_input[COCOCommonAugBase.bboxes_index])
#        classes = np.array(classes if classes is not None else original_input[COCOCommonAugBase.classes_index])
#        base_information = np.array(original_input[COCOCommonAugBase.base_information_index])
#        classes = np.delete(classes, np.argwhere(np.isnan(bboxes)), axis=0)
#        bboxes = np.delete(bboxes, np.argwhere(np.isnan(bboxes)), axis=0)
#        return image, bboxes.tolist(), base_information, classes.tolist()
#
#    @staticmethod
#    def image(*args):
#        return args[COCOCommonAugBase.image_index]
#
#    @staticmethod
#    def bboxes(*args):
#        return args[COCOCommonAugBase.bboxes_index]
#    
#    @staticmethod
#    def classes(*args):
#        return args[COCOCommonAugBase.classes_index]
#
#    @staticmethod
#    def base_information(*args):
#        return args[COCOCommonAugBase.base_information_index]
#    pass
#
##pcd.CommonDataManager.register('COCO', COCO)