# coding=utf-8
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
import Putil.data.common_data as pcd
from pycocotools.coco import COCO

logger = plog.PutilLogConfig('coco').logger()
logger.setLevel(plog.DEBUG)
COCODataLogger = logger.getChild('COCOData')
COCODataLogger.setLevel(plog.DEBUG)

import Putil.data.vision_common_convert.bbox_convertor as bbox_convertor
from Putil.data.util.vision_util.detection_util import rect_angle_over_border as rect_angle_over_border
from Putil.data.util.vision_util.detection_util import clip_box_using_image as clip_box 


class COCOData(pcd.CommonDataWithAug):
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

    class Stage(Enum):
        STAGE_TRAIN = 0
        STAGE_EVAL = 1
        STAGE_TEST = 2
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
        use_rate=1.0,
        image_width=128,
        image_height=128):
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
         @param[in] use_rate
         data used rate
        '''
        pcd.CommonDataWithAug.__init__(self, use_rate=use_rate)
        self._coco_root_dir = coco_root_dir
        self._stage = stage
        self._img_root_name = 'train2017' if self._stage == COCOData.Stage.STAGE_TRAIN else \
            ('val2017' if self._stage == COCOData.Stage.STAGE_EVAL else 'test2017')
        self._img_root_dir = os.path.join(self._coco_root_dir, self._img_root_name)
        self._detection = detection
        self._key_points = key_points
        self._stuff = stuff
        self._panoptic = panoptic
        self._dense_pose = dense_pose
        self._captions = captions
        assert True in [self._detection, self._key_points, self._stuff, self._panoptic, self._dense_pose, self._captions]
        
        self._instances_file_train = os.path.join(self._coco_root_dir, 'annotations/instances_train2017.json')
        self._instances_file_eval = os.path.join(self._coco_root_dir, 'annotations/instances_val2017.json')
        self._person_keypoints_train = os.path.join(self._coco_root_dir, 'annotations/person_keypoints_train2017.json')
        self._person_keypoints_eval = os.path.join(self._coco_root_dir, 'annotations/person_keypoints_val2017.json')
        self._captions_train = os.path.join(self._coco_root_dir, 'annotations/captions_train2017.json')
        self._captions_eval = os.path.join(self._coco_root_dir, 'annotations/captions_val2017.json')
        self._image_info_test = os.path.join(self._coco_root_dir, 'annotations/image_info_test2017.json')

        belong_instances = [self._detection, self._stuff, self._panoptic]
        belong_person_keypoints = [self._key_points]
        belong_captions = [self._captions]

        with_label = [COCOData.Stage.STAGE_TRAIN, COCOData.Stage.STAGE_EVAL]
        without_label = [COCOData.Stage.STAGE_TEST]
        self._instances_coco, instances_load = (COCO(self._instances_file_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._instances_file_eval), True) \
                if ((self._stage in with_label) and (True in [self._detection, self._stuff, self._panoptic])) else (None, False)
        self._instances_img_ids = self._instances_coco.getImgIds() if instances_load else list() 
        self._person_keypoints_coco, key_point_load = (COCO(self._person_keypoints_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._person_keypoints_eval), True) \
                if ((self._stage in with_label) and (self._key_points)) else (None, False)
        self._persion_keypoints_img_ids = self._person_keypoints_coco.getImgIds() if key_point_load else list()
        self._captions_coco, captions_load = (COCO(self._captions_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._captions_eval), True) \
                if ((self._stage in with_label) and (self._captions)) else (None, False)
        self._captions_img_ids = self._captions_coco.getImgIds() if captions_load else list()
        self._image_test, image_test_load = (COCO(self._image_info_test), True) if self._stage in without_label else (None, False)
        self._image_test_img_ids = self._image_test.getImgIds() if image_test_load else list()

        assert [instances_load, key_point_load, captions_load, image_test_load].count(True) == 1, "only support one ann file"

        # we know use the detectio only
        #self._data_field = COCOData.__get_common_id([self._instances_img_ids, self._persion_keypoints_img_ids, \
        #     self._captions_img_ids, self._image_test_img_ids])
        self._data_field = self._instances_img_ids + self._persion_keypoints_img_ids + self._captions_img_ids + self._image_test_img_ids
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

        if self._instances_coco is not None:
            self._instances_category_ids = self._instances_coco.getCatIds()
            pass

        self._image_width = image_width
        self._image_height = image_height
        pass

    def _restart_process(self, restart_param):
        self._image_width = restart_param('image_width', self._image_width)
        self._image_height = restart_param.get('image_height', self._image_height)
        pass

    def _inject_operation(self, inject_param):
        pass

    def __read_image(self, file_name):

        image = cv2.imread(os.path.join(self._img_root_dir, file_name)).astype(np.float32)
        image_min = np.min(np.min(image, axis=0, keepdims=True), axis=1, keepdims=True)
        image_max = np.max(np.max(image, axis=0, keepdims=True), axis=1, keepdims=True)
        image = (image - image_min) / (image_max - image_min)
        assert(image is not None)
        return image
        pass

    def _generate_from_origin_index(self, index):
        '''
         @brief generate the image [detection_label ]
         @note
         @ret 
         image [height, width, channel] numpy.float32
         bboxes 
        '''
        if self._stage == COCOData.Stage.STAGE_TEST:
            return self.__generate_test_from_origin_index(index)
        elif True in [self._detection, self._stuff, self._panoptic]:
            image_ann = self._instances_coco.loadImgs(self._data_field[index])
            ann_ids = self._instances_coco.getAnnIds(self._data_field[index])
            anns = self._instances_coco.loadAnns(ann_ids)

            image = self.__read_image(image_ann[0]['file_name'])

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
                box = ann['bbox']
                classes.append(self._instances_category_ids.index(ann['category_id']))
                #bboxes.append([(box[0] + 0.5 * box[2]) * x_scale, (box[1] + 0.5 * box[3]) * y_scale, box[2] * x_scale, box[3] * y_scale])
                bboxes.append([box[0] * x_scale, box[1] * y_scale, box[2] * x_scale, box[3] * y_scale])
                pass
            #for box in bboxes:
            #    cv2.rectangle(image, (box[0] - box[])
            #assert rect_angle_over_border(bboxes, image.shape[1], image.shape[0]) is False, "cross the border"
            if index == 823:
                pass
            bboxes = clip_box(bboxes, image)
            COCODataLogger.debug('original check:')
            ret = COCOCommonAugBase._repack(image, bboxes, classes, image=image, bboxes=bboxes, classes=classes)
            ret = self._aug_check(*ret)
            _bboxes = COCOCommonAugBase.bboxes(*ret)
            if len(_bboxes) == 0:
                COCODataLogger.warning('original data generate no obj, regenerate')
                ret = self._generate_from_specified(random.choice(range(0, len(self))))
                pass
            return ret
        else:
            raise NotImplementedError('unimplemented')
            pass
        pass

    def _aug_check(self, *args):
        if self._stage == COCOData.Stage.STAGE_TRAIN or (self._stage == COCOData.Stage.STAGE_EVAL):
            if True in [self._detection, self._stuff, self._panoptic]:
                bboxes = args[1]
                classes = args[2]
                assert len(bboxes) == len(classes)
                COCODataLogger.warning('zero obj occu') if len(bboxes) == 0 else None
                if len(bboxes) == 0:
                    pass
                assert np.argwhere(np.isnan(np.array(bboxes))).size == 0
                pass
            else:
                # TODO: other type
                pass
        elif self._stage == COCOData.Stage.STAGE_TEST:
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
        image = self.__read_image(image_ann[0]['file_name'])
        return image,

    def __generate_instance_from_origin_index(self, index):
        pass

    def __generate_keypoint_from_origin_index(self, index):
        pass

    def __generate_caption_from_origin_index(self, index):
        pass
    pass


pcd.CommonDataManager.register('COCOData', COCOData)


class COCOCommonAugBase:
    instance_image_index = 0
    image_index = instance_image_index
    instance_bboxes_index = 1
    bboxes_index = instance_bboxes_index
    instance_classes_index= 2
    class_index = instance_classes_index

    @staticmethod
    def _repack(*original_input, image=None, bboxes=None, classes=None):
        image = image if image is not None else original_input[COCOCommonAugBase.image_index]
        bboxes = np.array(bboxes if bboxes is not None else original_input[COCOCommonAugBase.bboxes_index])
        classes = np.array(classes if classes is not None else original_input[COCOCommonAugBase.classes_index])
        classes = np.delete(classes, np.argwhere(np.isnan(bboxes)), axis=0)
        bboxes = np.delete(bboxes, np.argwhere(np.isnan(bboxes)), axis=0)
        return image, bboxes.tolist(), classes.tolist()

    @staticmethod
    def image(*args):
        return args[COCOCommonAugBase.image_index]

    @staticmethod
    def bboxes(*args):
        return args[COCOCommonAugBase.bboxes_index]
    
    @staticmethod
    def classes(*args):
        return args[COCOCommonAugBase.classes_index]