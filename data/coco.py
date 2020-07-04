# coding=utf-8
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

    def __get_common_id(self, id_lists):
        if len(id_lists) > 1:
            common_list = list()
            for sample in id_lists[0]:
                view = [sample in id_list for id_list in id_lists[1:]]
                common_list.append(sample) if False no in view else None
                pass
            return common_list
            pass
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
        self._coco_root_dir = coco_root_dir
        self._stage = stage
        self._detection = detection
        self._key_points = key_points
        self._stuff = stuff
        self._panoptic = panoptic
        self._dense_pose = dense_pose
        self._captions = captions
        
        self._instances_file_train = os.path.join(self._coco_root_dir, 'annotations/instances_train2017.json')
        self._instances_file_eval = os.path.join(self._coco_root_dir, 'annotations/instances_val2017.json')
        self._person_keypoints_train = os.path.join(self._coco_root_dir, 'annotations/person_keypoint_train2017.json')
        self._person_keypoints_eval = os.path.join(self._coco_root_dir, 'annotations/person_keypoint_val2017.json')
        self._captions_train = os.path.join(self._coco_root_dir, 'annotations/captions_train2017.json')
        self._captions_eval = os.path.join(self._coco_root_dir, 'annotations/captions_val2017.json')
        self._image_info_test = os.path.join(self._coco_root_dir, 'annotations/image_info_test2017.json')

        with_label = [COCOData.Stage.STAGE_TRAIN, COCOData.Stage.STAGE_EVAL]
        without_label = [COCOData.Stage.STAGE_TEST]
        self._instances_coco = COCO(self._instances_file_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._instances_file_eval) \
                if self._stage in with_label else None
        self._instances_img_ids = self._instances_coco.getImgIds()
        self._person_keypoints_coco = COCO(self._person_keypoints_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._person_keypoints_eval) \
                if self._stage in with_label else None
        self._persion_keypoints_img_ids = self._person_keypoints_coco.getImgIds()
        self._captions_coco = COCO(self._captions_train \
            if self._stage == COCOData.Stage.STAGE_TRAIN else self._captions_eval) \
                if self._stage in with_label else None
        self._captions_img_ids = self._captions_coco.getImgIds()
        self._image_test = COCO(self._image_info_test) if self._stage in without_label else None
        self._image_test_img_ids = self._captions_coco.getImgIds()
        pass

    def _restart_process(self, restart_param):
        '''
        '''
        pass

    def _inject_operation(self, inject_param):
        pass

    def _generate_from_origin_index(self, index):
        '''
         @brief generate the image [detection_label ]
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
#In[]:
import json
person_file = '/data2/Public_Data/COCO/annotations/person_keypoints_val2017.json'
instance_file = '/data2/Public_Data/COCO/annotations/instances_val2017.json'
image_file = '/data2/Public_Data/COCO/annotations/image_info_test2017.json'
captions_file = '/data2/Public_Data/COCO/annotations/captions_val2017.json'
panno = json.loads(open(person_file, 'r').read())
print(panno.keys())
ianno = json.loads(open(instance_file, 'r').read())
print(ianno.keys())
imanno = json.loads(open(image_file, 'r').read())
print(imanno.keys())
canno = json.loads(open(captions_file, 'r').read())
print(canno.keys())
#In[]:
canno['annotations']
#In[]:
from pycocotools.coco import COCO
ic = COCO(instance_file)
len(ic.getImgIds())
#In[]:
from pycocotools.coco import COCO
cc = COCO(captions_file)
len(cc.getImgIds())
#In[]:
from pycocotools.coco import COCO
cc = COCO(person_file)
len(cc.getImgIds())
#In[]:
import json
data_root = '/data2/Public_Data/COCO/train2017'
person_file = '/data2/Public_Data/COCO/annotations/person_keypoints_train2017.json'
instance_file = '/data2/Public_Data/COCO/annotations/instances_train2017.json'
captions_file = '/data2/Public_Data/COCO/annotations/captions_train2017.json'
#panno = json.loads(open(person_file, 'r').read())
#print(panno.keys())
#ianno = json.loads(open(instance_file, 'r').read())
#print(ianno.keys())
#imanno = json.loads(open(image_file, 'r').read())
#print(imanno.keys())
#canno = json.loads(open(captions_file, 'r').read())
#print(canno.keys())
#In[]:
canno['annotations']
#In[]:
from pycocotools.coco import COCO
ic = COCO(instance_file)
#In[]:
from skimage import io
from matplotlib import pyplot as plt
import os
imgids = ic.getImgIds()
img = ic.loadImgs(imgids[0:1])
print('imageid len: {0}'.format(len(imgids)))
labids = ic.getAnnIds(imgids[0:1])
print('label len: {0} for image: {1}'.format(len(labids), imgids[0:1]))
labels = ic.loadAnns(labids)
I = io.imread(os.path.join(data_root, img[0]['file_name']))
plt.axis('off')
plt.imshow(I) #绘制图像，显示交给plt.show()处理
ic.showAnns(labels, draw_bbox=False)
plt.show()
#In[]:
from pycocotools.coco import COCO
cc = COCO(captions_file)
len(cc.getImgIds())
#In[]:
from pycocotools.coco import COCO
cc = COCO(person_file)
len(cc.getImgIds())
#In[]:
import time
begin = time.clock()
end = time.clock()
print(begin)
print(end)