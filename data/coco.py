# coding=utf-8
import copy
from colorama import Fore
#from pycocotools.cocoeval import COCOeval
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
from importlib import reload
import cv2
import time
from enum import Enum
from PIL import Image
import os
import json
import numpy as np
from pycocotools.coco import COCO
import Putil.base.logger as plog

logger = plog.PutilLogConfig('coco').logger()
logger.setLevel(plog.DEBUG)
COCODataLogger = logger.getChild('COCOData')
COCODataLogger.setLevel(plog.DEBUG)
COCOBaseLogger = logger.getChild('COCOBase')
COCOBaseLogger.setLevel(plog.DEBUG)

from Putil.data import cocoeval
reload(cocoeval)
COCOeval = cocoeval.CustomCOCOeval
import Putil.data.vision_common_convert.bbox_convertor as bbox_convertor
reload(bbox_convertor)
from Putil.data.util.vision_util.detection_util import rect_angle_over_border as rect_angle_over_border
from Putil.data.util.vision_util import detection_util
rect_angle_over_border = detection_util.rect_angle_over_border
clip_box = detection_util.clip_box_using_image
reload(detection_util)
rect_angle_over_border = detection_util.rect_angle_over_border
clip_box = detection_util.clip_box_using_image
import Putil.data.common_data as pcd
reload(pcd)


class COCOBase(pcd.CommonDataForTrainEvalTest):
    '''
     @brief
     @note
        有关coco的信息，总共有四类大任务：目标检测detection、全景分割panoptic、图像内容描述captions、人体目标点检测keypoints
        使用getImgIds不指定catIds获取到的img_ids是所有图片的id，可以使用[v['image_id'] for k, v in coco.anns.items()]来获取
    真正对应有目标任务ann的图片的id，通过coco_basical_statistic可以得知：三个标注文件是包含关系caption包含instance包含person_keypoint
    '''
    # represent->cat_id->cat_name->represent
    represent_to_cat_id = OrderedDict({0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46, 41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90})
    cat_id_to_represent = OrderedDict()
    for represent, cat_id in represent_to_cat_id.items():
        cat_id_to_represent[cat_id] = represent
    #cat_id_to_represent = {cat_id: represent for represent, cat_id in represent_to_cat_id.items()}
    cat_id_to_cat_name = OrderedDict({1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light', 11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'})
    cat_name_to_represent = OrderedDict()
    for cat_id, cat_name in cat_id_to_cat_name.items():
        cat_name_to_represent[cat_name] = cat_id_to_represent[cat_id]
    # TODO: important problem remain 当使用以下方法生成cat_name_to_represent时出现cat_id_to_represent undefined的情况
    #cat_name_to_represent = {cat_name: cat_id_to_represent[cat_id] for cat_id, cat_name in cat_id_to_cat_name.items()}
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
    result_image_index = 1 # image 表示当前的图像，shape为[Height, Width, Channel], 类型为RGB，注意cv2读取以及写图像都默认是BGR格式
    result_detection_box_index = 2 # format: [[top_x, top_y, width, height], ...]
    result_detection_class_index = 3 # format: [class_represent] class_represent表示的是当前class使用的索引号，不是cat_id
    result_detection_score_index = 4

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
        return COCOBase.represent_to_cat_id[COCOBase.cat_name_to_represent[cat_name]] if cat_name is not None else COCOBase.represent_to_cat_id[represent_value]

    @staticmethod
    def detection_get_cat_name(cat_id=None, represent_value=None):
        assert False in [t is None for t in [cat_id, represent_value]]
        return COCOBase.cat_id_to_cat_name[cat_id] if cat_id is not None else COCOBase.cat_id_to_cat_name[COCOBase.represent_to_cat_id[represent_value]]
    
    @staticmethod
    def coco_basical_statistic(coco_root_dir, save_to):
        '''
         @brief
           统计每个任务形式使用的图像数量以及重叠数量
        '''
        instances_file_train = os.path.join(coco_root_dir, 'annotations/instances_train2017.json')
        instances_file_eval = os.path.join(coco_root_dir, 'annotations/instances_val2017.json')
        person_keypoints_train = os.path.join(coco_root_dir, 'annotations/person_keypoints_train2017.json')
        person_keypoints_eval = os.path.join(coco_root_dir, 'annotations/person_keypoints_val2017.json')
        captions_train = os.path.join(coco_root_dir, 'annotations/captions_train2017.json')
        captions_eval = os.path.join(coco_root_dir, 'annotations/captions_val2017.json')
        image_info_test = os.path.join(coco_root_dir, 'annotations/image_info_test2017.json')
        # img_amount
        result = list()
        itcoco = COCO(instances_file_train)
        i_train_img_ids = set([v['image_id'] for k, v in itcoco.anns.items()])
        result.append({'name': 'train_instance', 'img_amount': len(i_train_img_ids)})
        ptcoco = COCO(person_keypoints_train)
        p_train_img_ids = set([v['image_id'] for k, v in ptcoco.anns.items()])
        result.append({'name': 'train_person_keypoint', 'img_amount': len(p_train_img_ids)})
        ctcoco = COCO(captions_train)
        c_train_img_ids = set([v['image_id'] for k, v in ctcoco.anns.items()])
        result.append({'name': 'train_caption', 'img_amount': len(c_train_img_ids)})
        img_ids_in_instance_and_person_keypoint = [i for i in p_train_img_ids if i in i_train_img_ids]
        result.append({'name': 'train_instance_person_keypoint', 'img_amount': len(img_ids_in_instance_and_person_keypoint)})
        img_ids_in_instance_and_caption = [i for i in p_train_img_ids if i in c_train_img_ids]
        result.append({'name': 'train_instance_caption', 'img_amount': len(img_ids_in_instance_and_caption)})
        img_ids_in_instance_and_person_keypoint_and_caption = [i for i in img_ids_in_instance_and_person_keypoint if i in img_ids_in_instance_and_caption]
        result.append({'name': 'instance_person_keypoint_caption', 'img_amount': len(img_ids_in_instance_and_person_keypoint_and_caption)})

        iecoco = COCO(instances_file_eval)
        i_eval_img_ids = set([v['image_id'] for k, v in iecoco.anns.items()])
        result.append({'name': 'eval_instance', 'img_amount': len(i_eval_img_ids)})
        pecoco = COCO(person_keypoints_eval)
        p_eval_img_ids = set([v['image_id'] for k, v in pecoco.anns.items()])
        result.append({'name': 'eval_person_keypoint', 'img_amount': len(p_eval_img_ids)})
        cecoco = COCO(captions_eval)
        c_eval_img_ids = set([v['image_id'] for k, v in cecoco.anns.items()])
        result.append({'name': 'eval_caption', 'img_amount': len(c_eval_img_ids)})
        img_ids_in_instance_and_person_keypoint = [i for i in p_eval_img_ids if i in i_eval_img_ids]
        result.append({'name': 'eval_instance_person_keypoint', 'img_amount': len(img_ids_in_instance_and_person_keypoint)})
        img_ids_in_instance_and_caption = [i for i in p_eval_img_ids if i in c_eval_img_ids]
        result.append({'name': 'eval_instance_caption', 'img_amount': len(img_ids_in_instance_and_caption)})
        img_ids_in_instance_and_person_keypoint_and_caption = [i for i in img_ids_in_instance_and_person_keypoint if i in img_ids_in_instance_and_caption]
        result.append({'name': 'instance_person_keypoint_caption', 'img_amount': len(img_ids_in_instance_and_person_keypoint_and_caption)})
        result_df = pd.DataFrame(result)
        plt.rcParams['savefig.dpi'] = 300
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, frame_on=False) # no visible frame
        ax.xaxis.set_visible(False)  # hide the x axis
        ax.yaxis.set_visible(False)  # hide the y axis
        table(ax, result_df, loc='center')
        plt.savefig(os.path.join(save_to, 'basical_statistic.png'))
        pass

    @staticmethod
    def detection_statistic_obj_size_follow_cat(cat_names, ann_file, save_to):
        cat_ids = [COCOBase.represent_to_cat_id[COCOBase.cat_name_to_represent[cat_name]] for cat_name in cat_names] if type(cat_names).__name__ == 'list'\
            else [COCOBase.represent_to_cat_id[COCOBase.cat_name_to_represent[cat_names]]]
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
            plt.title(COCOBase.cat_id_to_cat_name[cat_id])
            plt.ylabel('Counts')
            plt.xlabel('bbox area/100')
            plt.savefig(os.path.join(save_to, 'box_area_histogram_{}.png'.format(COCOBase.cat_id_to_cat_name[cat_id])))
            plt.close()
            #plt.show()
            #hist, xedges, yedges = np.histogram2d(anns_df['bbox'].apply(lambda x: x[2]), anns_df['bbox'].apply(lambda x: x[3]), bins=1000)
            pass
        pass

    @staticmethod
    def detection_statistic_img_amount_obj_amount(ann_file, save_to, cat_name=None):
        coco = COCO(ann_file)
        if cat_name is not None:
            cat_ids = [COCOBase.represent_to_cat_id[COCOBase.cat_name_to_represent[cat_name]] for cat_name in cat_names] if type(cat_names).__name__ == 'list'\
                else [COCOBase.represent_to_cat_id[COCOBase.cat_name_to_represent[cat_names]]]
            pass
        else:
            cat_ids = coco.getCatIds()
            pass
        result = list()
        for cat_id in cat_ids:
            img_id = coco.getImgIds(catIds=[cat_id])
            ann_id = coco.getAnnIds(catIds=[cat_id])
            result.append({'category': COCOBase.cat_id_to_cat_name[cat_id], 'img_amount': len(img_id), \
                'cat_id': cat_id, 'obj_amount': len(ann_id)})
            pass
        result.append({'category': 'all', 'img_amount': len(set([v['image_id'] for k, v in coco.anns.items()])), 'cat_id': 'all', \
            'obj_amount': len(coco.anns)})
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
        use_rate,
        remain_strategy,
        cat_ids=None,
    ):
        pcd.CommonDataWithAug.__init__(self, use_rate=use_rate, sub_data=cat_ids, remain_strategy=remain_strategy)
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
        self._person_keypoints_file_train = os.path.join(self._coco_root_dir, 'annotations/person_keypoints_train2017.json')
        self._person_keypoints_file_eval = os.path.join(self._coco_root_dir, 'annotations/person_keypoints_val2017.json')
        self._captions_file_train = os.path.join(self._coco_root_dir, 'annotations/captions_train2017.json')
        self._captions_file_eval = os.path.join(self._coco_root_dir, 'annotations/captions_val2017.json')
        self._image_info_test = os.path.join(self._coco_root_dir, 'annotations/image_info_test2017.json')

        belong_instances = [self._detection, self._stuff, self._panoptic]
        belong_person_keypoints = [self._key_points]
        belong_captions = [self._captions]

        with_label = [COCOBase.Stage.Train, COCOBase.Stage.Evaluate]
        without_label = [COCOBase.Stage.Test]
        self._captions_coco, captions_load = (COCO(self._captions_file_train \
            if self._stage == COCOBase.Stage.Train else self._captions_file_eval), True) \
                if ((self._stage in with_label) and (self._captions)) else (None, False)
        self._captions_img_ids = list(set([v['image_id'] for k, v in self._captions_coco.anns.items()])) if captions_load else list()
        self._instances_coco, instances_load = (COCO(self._instances_file_train \
            if self._stage == COCOBase.Stage.Train else self._instances_file_eval), True) \
                if ((self._stage in with_label) and (True in [self._detection, self._stuff, self._panoptic])) else (None, False)
        self._instances_img_ids = list(set([v['image_id'] for k, v in self._instances_coco.anns.items()])) if instances_load else list() 
        self._person_keypoints_coco, key_point_load = (COCO(self._person_keypoint_file_train \
            if self._stage == COCOBase.Stage.Train else self._person_keypoints_file_eval), True) \
                if ((self._stage in with_label) and (self._key_points)) else (None, False)
        self._person_keypoints_img_ids = list(set([v['image_id'] for k, v in self._preson_keypoints_coco.anns.items()])) if key_point_load else list()

        self._image_test, image_test_load = (COCO(self._image_info_test), True) if self._stage in without_label else (None, False)
        self._image_test_img_ids = self._image_test.getImgIds() if image_test_load else list()

        assert [instances_load, key_point_load, captions_load, image_test_load].count(True) == 1, 'only support one type'
        #COCOBaseLogger.warning('') if self._cat_id != COCOBase.
        # we know use the detectio only
        #self._data_field = COCOBase.__get_common_id([self._instances_img_ids, self._persion_keypoints_img_ids, \
        #     self._captions_img_ids, self._image_test_img_ids])
        # TODO:record
        if self._stage in [COCOBase.Stage.Train, COCOBase.Stage.Evaluate]:
            if key_point_load:
                assert not instances_load and not captions_load, 'should not use person_keypoint with caption and instance'
                COCOBaseLogger.warning(Fore.RED + 'cat_id is invalid in person_keypoint' + Fore.RESET) if self._cat_ids is not None else None
                self._data_field = self._person_keypoints_img_ids
            if instances_load:
                COCOBaseLogger.info('use instance{}'.format(' and caption' if captions_load else ''))
                self._data_field = self._instances_coco.getImgIds(catIds=self._cat_ids) if self._cat_ids is not None and self._remain_strategy == COCOBase.RemainStrategy.Drop else self._instances_img_ids
                self._cat_id_to_represent = copy.deepcopy(COCOBase.cat_id_to_represent) if self._cat_ids is None else {cat_id: index for index, cat_id in enumerate(self._cat_ids)}
                self._represent_to_cat_id = {v: k for k, v in self._cat_id_to_represent.items()}
                if self._information_save_to_path is not None:
                    with open(os.path.join(self._information_save_to_path, 'detection_cat_id_to_represent.json'), 'w') as fp:
                        json.dump(self._cat_id_to_represent, fp, indent=4)
            if captions_load and not instances_load:
                COCOBaseLogger.info('use caption')
                self._data_field = self._captions_img_ids
        elif self._stage == COCOBase.Stage.Test:
            COCOBaseLogger.warning(Fore.RED + 'cat_ids is invalid in Test' + Fore.RESET) if self._cat_ids is not None else None
            self._data_field = self._image_test_img_ids
        else:
            raise NotImplementedError(Fore.RED + 'Stage: {} is not Implemented'.format(stage) + Fore.RESET)
        self._fix_field()
        ## check the ann
        #if self._stage in with_label:
        #    image_without_ann = dict()
        #    for index in self._data_field:
        #        image_ann = self._instances_coco.loadImgs(index)
        #        ann_ids = self._instances_coco.getAnnIds(index)
        #        if len(ann_ids) == 0:
        #            image_without_ann[index] = image_ann
        #    for index_out in list(image_without_ann.keys()):
        #        self._data_field.remove(index_out)
        #    with open('./image_without_ann.json', 'w') as fp:
        #        str_ = json.dumps(image_without_ann, indent=4)
        #        fp.write(str_)
        #        pass
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

    def save_result(self, result=None, save=False, prefix=None):
        '''
         @brief
         @note
          输入的result以image为基准，每一个result包含一个image，所有任务公共结果信息为image与base_information
          detection任务的result说明：
            主要bboxes，classes，scores
                bboxes, ndarray, 为依照datas输出image的尺寸的bboxes，单个bbox格式为[top_x,top_y,width,height]，一张图多个bbox使用list组成bboxes
                classes, ndarray, 使用的是模型输出的classes，COCO中存在cat_id<-->represent的映射关系，result中的classes使用的是represent，这样有利于
            COCO的封闭性与完整性，单个classes的格式为：int，一张图每个bbox对应一个class，使用list组成classes
                scores, ndarray, 使用的是模型输出的score，float，每个bbox对应一个score，使用list组成scores
         @param[in] result result默认为None，当result为None时，是不增加result数据的，其他照常进行
         @param[in] save bool类型，决定是否将当前的result保存到文件中
        '''
        if result is None:
            self.add_detection_result(save=save, prefix=prefix)
            return
        if self._detection:
            base_information = result[COCOBase.result_base_information_index]
            image = result[COCOBase.result_image_index]
            image_id = base_information[COCOBase.image_id_index_in_base_information]
            image_width = base_information[COCOBase.image_width_index_in_base_information]
            image_height = base_information[COCOBase.image_height_index_in_base_information]
            # true_image_size / resized_image_size = true_box_size / resized_box_size
            bboxes = result[COCOBase.result_detection_box_index] * ([image_width / image.shape[1], image_height / image.shape[0]] * 2)
            bboxes = bboxes if type(bboxes).__name__ == 'list' else bboxes.tolist()
            classes = result[COCOBase.result_detection_class_index]
            classes = [self._represent_to_cat_id[_class] for _class in classes]
            classes = classes if type(classes).__name__ == 'list' else classes.tolist()
            scores = result[COCOBase.result_detection_score_index]
            scores = scores if type(scores).__name__ == 'list' else scores.tolist()
            self.add_detection_result(
                image=image,
                image_id=image_id,
                category_ids=classes, bboxes=bboxes, scores=scores, save=save, prefix=prefix)
            return 0
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
        sync_status = [image is None, image_id is None, category_ids is None, bboxes is None, scores is None]
        if True in sync_status:
            COCODataLogger.warning(Fore.RED + 'None found in the result data, nothing would be add to result' + Fore.RESET)
        else:
            used_wh = image.shape[0: 2][::-1]
            self._detection_result = pd.DataFrame() if self._detection_result is None else self._detection_result
            result_temp = list()
            for category_id, bbox, score in zip(category_ids, bboxes, scores):
                result_temp.append({'image_id': image_id, 'category_id': category_id, 'bbox': bbox, 'score': score})
            self._detection_result = self._detection_result.append(result_temp, ignore_index=True)
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

    def evaluate(self, image_ids=None, cat_ids=None, prefix=None, use_visual=False):
        pass

    def evaluate_detection(self, image_ids=None, cat_ids=None, scores=None, ious=None, prefix=None, use_visual=False):
        '''
         @brief evaluate the performance
         @note use the result files in the self._information_save_to_path, combine all result files and save to a json file, and
         then we would use this json file to evaluate the performance, base on object the image_ids Cap cat_ids
         @param[in] image_ids the images would be considered in the evaluate, 当没有指定时，则对目标coco的getImgIds的所有image进行evaluate
         @param[in] cat_ids the categories would be considered in the evaluate，当没有指定时，则对目标coco的getCatIds的所有cat进行evaluate
         @param[in] scores list格式，阈值，超过该值的bbox才被考虑
         @param[in] ious list格式，阈值，考虑基于这些iou阈值的ap与ar
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
        cat_ids = cat_ids if cat_ids is not None else self._instances_coco.getCatIds()
        if scores is not None:
            t = lambda x, score: x >= score
        else:
            t = lambda x, score: x != None
        scores = [None] if scores is None else scores
        for score in scores:
            sub_detection_result = detection_result[t(detection_result['score'], score)]
            if use_visual:
                visual_save_to = os.path.join(self._information_save_to_path, '{}-{}'.format(prefix, score))
                if os.path.exists(visual_save_to) and os.path.isdir(visual_save_to):
                    pass
                else:
                    os.mkdir(visual_save_to)
                    pass
                from Putil.trainer.visual.image_visual.point_visual import PointVisual
                from Putil.trainer.visual.image_visual.rectangle_visual import RectangleVisual
                pv = PointVisual(); rv = RectangleVisual(2)
                image_ids = self._instances_coco.getImgids() if image_ids is None else image_ids
                img_anns = self._instances_coco.loadImgs(image_ids)
                for image_id in image_ids:
                    img_ann = self._instances_coco.loadImgs([image_id])[0]
                    img_numpy = self.read_image(img_ann['file_name'])
                    result_for_this_image = sub_detection_result[sub_detection_result['image_id']==img_ann['id']]
                    def return_center_xy(s):
                        '''提供生成中心点x，y的方法，用于DataFrame的apply'''
                        s['x'] = s['bbox'][0] + 0.5 * s['bbox'][2]
                        s['y'] = s['bbox'][1] + 0.5 * s['bbox'][3]
                        return s
                    def return_normal_rectangle(s):
                        '''提供分离bbox的方法，因为原本在DataFrame中存储bbox使用的是一个list，没法转化为[*, 4]格式的ndarry
                        目前没有找到其他方法，使用该函数分离[top_x, top_y, width, height]'''
                        s['top_x'], s['top_y'], s['w'], s['h'] = s['bbox'][0], s['bbox'][1], s['bbox'][2], s['bbox'][3]
                        return s
                    labels_for_this_image = self._instances_coco.loadAnns(self._instances_coco.getAnnIds(imgIds=[image_id], catIds=cat_ids))
                    if not result_for_this_image.empty:
                        # visual the pre
                        img_visual = pv.visual_index(img_numpy, result_for_this_image.apply(return_center_xy, axis=1)[['x', 'y']].values, [0, 255, 0])
                        img_visual = rv.rectangle_visual(img_visual, pd.DataFrame(result_for_this_image['bbox']).apply(return_normal_rectangle, axis=1)[['top_x', 'top_y', 'w', 'h']].values, \
                            scores=result_for_this_image['score'], fontScale=0.3)
                    else:
                        img_visual = img_numpy
                    if len(labels_for_this_image) != 0:
                        # visual the gt
                        gt_bboxes = np.array([label['bbox'] for label in labels_for_this_image])
                        gt_center_xy = gt_bboxes[:, 0: 2] + gt_bboxes[:, 2: 4] / 2.0
                        img_visual = pv.visual_index(img_visual, gt_center_xy, [255, 0, 0])
                        img_visual = rv.rectangle_visual(img_visual, gt_bboxes, fontScale=0.3, color_map=[[255, 0, 0]])
                    cv2.imwrite(os.path.join(visual_save_to, '{}.png'.format(img_ann['id'])), cv2.cvtColor(img_visual, cv2.COLOR_RGB2BGR))
                pass
            index_name = {index: name for index, name in enumerate(list(sub_detection_result.columns))}
            sub_detection_result_formated = [{index_name[index]: tt for index, tt in enumerate(t)} for t in list(np.array(sub_detection_result))]
                
            json_file_path = os.path.join(self._information_save_to_path, '{}_score_{}_formated_sub_detection_result.json'.format(prefix, score))
            with open(json_file_path, 'w') as fp:
                json.dump(sub_detection_result_formated, fp)
        
            sub_detection_result_coco = self._instances_coco.loadRes(json_file_path)
            #result_image_ids = sub_detection_result_coco.getImgIds()
            cocoEval = COCOeval(self._instances_coco, sub_detection_result_coco, 'bbox')
            cocoEval.params.imgIds  = image_ids if image_ids is not None else cocoEval.params.imgIds
            cocoEval.params.catIds = cat_ids if cat_ids is not None else cocoEval.params.catIds
            cocoEval.params.iouThrs = ious if ious is not None else cocoEval.params.iouThrs
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


class COCOData(COCOBase):
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
            key_points, stuff, panoptic, dense_pose, captions, use_rate, remain_strategy, cat_ids)
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
            #for ann in anns:
            #    box = ann['bbox']
            #    if (box[0] + box[2] > image.shape[1]) or (box[1] + box[3] > image.shape[0]):
            #        COCODataLogger.info(box)
            #        pass
            #    pass
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
                classes.append(self._cat_id_to_represent[ann['category_id']])
                #bboxes.append([(box[0] + 0.5 * box[2]) * x_scale, (box[1] + 0.5 * box[3]) * y_scale, box[2] * x_scale, box[3] * y_scale])
                bboxes.append([box[0] * x_scale, box[1] * y_scale, box[2] * x_scale, box[3] * y_scale])
                pass
            #for box in bboxes:
            #    cv2.rectangle(image, (box[0] - box[])
            #assert rect_angle_over_border(bboxes, image.shape[1], image.shape[0]) is False, "cross the border"
            #if index == 823:
            #    pass
            if len(bboxes) != 0:
                bboxes = clip_box(bboxes, image)
                classes = np.delete(classes, np.argwhere(np.isnan(bboxes)), axis=0)
                bboxes = np.delete(bboxes, np.argwhere(np.isnan(bboxes)), axis=0)
            datas[COCOBase.base_information_index] = base_information
            datas[COCOBase.image_index] = image
            datas[COCOBase.detection_box_index] = bboxes
            datas[COCOBase.detection_class_index] = classes
            #ret = self._aug_check(*ret)
            COCODataLogger.warning('original data generate no obj') if len(datas[COCOBase.detection_box_index]) == 0 else None
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