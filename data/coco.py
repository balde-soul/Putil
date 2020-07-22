# coding=utf-8
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
        captions=False):
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
        '''
        pcd.CommonDataWithAug.__init__(self)
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
        self._instances_img_ids = self._instances_coco.getImgIds() if instances_load else None
        self._person_keypoints_coco, key_point_load = (COCO(self._person_keypoints_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._person_keypoints_eval), True) \
                if ((self._stage in with_label) and (self._key_points)) else (None, False)
        self._persion_keypoints_img_ids = self._person_keypoints_coco.getImgIds() if key_point_load else None
        self._captions_coco, captions_load = (COCO(self._captions_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._captions_eval), True) \
                if ((self._stage in with_label) and (self._captions)) else (None, False)
        self._captions_img_ids = self._captions_coco.getImgIds() if captions_load else None
        self._image_test, image_test_load = (COCO(self._image_info_test), True) if self._stage in without_label else (None, False)
        self._image_test_img_ids = self._captions_coco.getImgIds() if image_test_load else None

        # we know use the detectio only
        self._index = COCOData.__get_common_id([self._instances_img_ids])
        
        # check the ann
        image_without_ann = dict()
        for index in self._index:
            image_ann = self._instances_coco.loadImgs(index)
            ann_ids = self._instances_coco.getAnnIds(index)
            if len(ann_ids) == 0:
                image_without_ann[index] = image_ann
        for index_out in list(image_without_ann.keys()):
            self._index.remove(index_out)
        with open('./image_without_ann.json', 'w') as fp:
            str_ = json.dumps(image_without_ann, indent=4)
            fp.write(str_)
            pass

        if self._instances_coco is not None:
            self._instances_category_ids = self._instances_coco.getCatIds()
        pass

    def _restart_process(self, restart_param):
        pass

    def _inject_operation(self, inject_param):
        pass

    def _generate_from_origin_index(self, index):
        '''
         @brief generate the image [detection_label ]
         @note
         @ret 
         (image, boxes)
        '''
        image_ann = self._instances_coco.loadImgs(self._index[index])
        ann_ids = self._instances_coco.getAnnIds(self._index[index])
        anns = self._instances_coco.loadAnns(ann_ids)

        image = cv2.imread(os.path.join(self._img_root_dir, image_ann[0]['file_name'])).astype(np.float32)
        image_min = np.min(np.min(image, axis=0, keepdims=True), axis=1, keepdims=True)
        image_max = np.max(np.max(image, axis=0, keepdims=True), axis=1, keepdims=True)
        image = (image - image_min) / (image_max - image_min)
        assert(image is not None)

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

        resize_width = 128
        resize_height = 128 
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
            print('break')
        bboxes = clip_box(bboxes, image)
        return self._aug_check(*COCOCommonAugBase._repack(image, bboxes, classes, image=image, bboxes=bboxes, classes=classes))

    def _aug_check(self, *args):
        bboxes = args[1]
        classes = args[2]
        assert len(bboxes) == len(classes)
        COCODataLogger.warning('zero obj occu') if len(bboxes) == 0 else None
        assert np.argwhere(np.isnan(np.array(bboxes))).size == 0
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
    pass

pcd.CommonDataManager.register('COCOData', COCOData)


class COCOCommonAugBase:
    image_index = 0
    bboxes_index = 1
    classes_index= 2

    @staticmethod
    def _repack(*original_input, image=None, bboxes=None, classes=None):
        image = image if image is not None else original_input[COCOCommonAugBase.image_index]
        bboxes = np.array(bboxes if bboxes is not None else original_input[COCOCommonAugBase.bboxes_index])
        classes = np.array(classes if classes is not None else original_input[COCOCommonAugBase.classes_index])
        classes = np.delete(classes, np.argwhere(np.isnan(bboxes)), axis=0)
        bboxes = np.delete(bboxes, np.argwhere(np.isnan(bboxes)), axis=0)
        return image, bboxes.tolist(), classes.tolist()

    @staticmethod
    def image(args):
        return args[COCOCommonAugBase.image_index]

    @staticmethod
    def bboxes(args):
        return args[COCOCommonAugBase.bboxes_index]
    
    @staticmethod
    def classes(args):
        return args[COCOCommonAugBase.classes_index]

##In[]:
#import json
#person_file = '/data2/Public_Data/COCO/annotations/person_keypoints_val2017.json'
#instance_file = '/data2/Public_Data/COCO/annotations/instances_val2017.json'
#image_file = '/data2/Public_Data/COCO/annotations/image_info_test2017.json'
#captions_file = '/data2/Public_Data/COCO/annotations/captions_val2017.json'
#panno = json.loads(open(person_file, 'r').read())
#print(panno.keys())
#ianno = json.loads(open(instance_file, 'r').read())
#print(ianno.keys())
#imanno = json.loads(open(image_file, 'r').read())
#print(imanno.keys())
#canno = json.loads(open(captions_file, 'r').read())
#print(canno.keys())
##In[]:
#canno['annotations']
##In[]:
#from pycocotools.coco import COCO
#ic = COCO(instance_file)
#len(ic.getImgIds())
##In[]:
#from pycocotools.coco import COCO
#cc = COCO(captions_file)
#len(cc.getImgIds())
##In[]:
#from pycocotools.coco import COCO
#cc = COCO(person_file)
#len(cc.getImgIds())
##In[]:
#import json
#data_root = '/data2/Public_Data/COCO/train2017'
#person_file = '/data2/Public_Data/COCO/annotations/person_keypoints_train2017.json'
#instance_file = '/data2/Public_Data/COCO/annotations/instances_train2017.json'
#captions_file = '/data2/Public_Data/COCO/annotations/captions_train2017.json'
##panno = json.loads(open(person_file, 'r').read())
##print(panno.keys())
##ianno = json.loads(open(instance_file, 'r').read())
##print(ianno.keys())
##imanno = json.loads(open(image_file, 'r').read())
##print(imanno.keys())
##canno = json.loads(open(captions_file, 'r').read())
##print(canno.keys())
##In[]:
#canno['annotations']
##In[]:
#from pycocotools.coco import COCO
#ic = COCO(instance_file)
##In[]:
#from skimage import io
#from matplotlib import pyplot as plt
#import os
#imgids = ic.getImgIds()
#img = ic.loadImgs(imgids[0:1])
#print('imageid len: {0}'.format(len(imgids)))
#labids = ic.getAnnIds(imgids[0:1])
#print('label len: {0} for image: {1}'.format(len(labids), imgids[0:1]))
#labels = ic.loadAnns(labids)
#print(labels[0])
#I = io.imread(os.path.join(data_root, img[0]['file_name']))
#plt.axis('off')
#plt.imshow(I) #绘制图像，显示交给plt.show()处理
#ic.showAnns(labels[0:1], draw_bbox=True)
#plt.show()
##In[]:
#from pycocotools.coco import COCO
#cc = COCO(captions_file)
#len(cc.getImgIds())
##In[]:
#from pycocotools.coco import COCO
#cc = COCO(person_file)
#len(cc.getImgIds())
##In[]:
#from PIL import Image
#import numpy as np
#import matplotlib.pyplot as plt
#import cv2
#
#print('array')
#arr = np.reshape(np.linspace(1, 9, num=9), [3, 3])
#print(arr)
#
#print('Image')
#img = Image.fromarray(arr)
#print(np.array(img))
#rimg = img.resize([6, 6], Image.BILINEAR)
#print(np.array(rimg))
#bimg = rimg.resize([3, 3], Image.BILINEAR)
#print(np.array(bimg))
#
#print('cv')
#rimg = cv2.resize(arr, (6, 6), interpolation=cv2.INTER_LINEAR)
#print(rimg)
#bimg = cv2.resize(rimg, (3, 3), interpolation=cv2.INTER_LINEAR)
#print(bimg)