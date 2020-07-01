# coding=utf-8
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


class COCOData(pcd.CommonData):
    @staticmethod
    def set_seed(seed):
        pass

    def __init__(self, captions_anno_file, information_save_to='', **kwargs):
        '''
        focus on the coco2017:
        the data is for Segmentation, object detection, personal key-point detection and pic to word
        the semantic label and bbox label is in the instances_*, and the label is not completed
        the key point label is in the person_keypoints_*
        the captions label is in the captions_*
        '''
        self._coco = COCO(anno_file)
        self._coco.imgs
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


class COCOStatistic:
    '''
    @brief
    '''
    pass

#In[]:

anno = json.loads(open(instance_file, 'r').read())

#In[]:
anno.keys()
#In[]:
anno['info']
#In[]:
import os
images = anno['images']
for i in images:
    if i['id'] == 448263:
        print(i)
        image_path = os.path.join('/data2/Public_Data/COCO/val2017', i['file_name'])
        break
    pass
import cv2
import matplotlib.pyplot as plt
img = cv2.imread(image_path)
plt.imshow(img)
plt.show()
#In[]:
annot = anno['annotations']
print(type(annot))
print(len(annot))
print(annot[0].keys())
annot[118]
for a in annot:
    if a['iscrowd'] == 0:
        print(a)
        break
        pass
for a in annot:
    if a['iscrowd'] == 1:
        print(a)
        break
        pass

#In[]:
person_file = '/data2/Public_Data/COCO/annotations/person_keypoints_val2017.json'
panno = json.loads(open(person_file, 'r').read())

#In[]:
panno.keys()

#In[]:
panno['categories']

#In[]:
panno_image = panno['images']
panno_image[0]

#In[]:
from pycocotools.coco import COCO
import json
instance_file = '/data2/Public_Data/COCO/annotations/instances_val2017.json'
iann = COCO(instance_file)
print(dir(iann))
#In[]:
import matplotlib.pyplot as plt
import numpy as np
ids = iann.getAnnIds(areaRng=[100, 200])
bbox = iann.loadAnns(ids[0])[0]['bbox']
area = iann.loadAnns(ids[0])[0]['area']
box_area = bbox[-1] * bbox[-2]
print('bbox area: {0}; area: {1}'.format(box_area, area))

iannc = iann.loadAnns(ids[0])
mask = iann.annToMask(iannc[0])
plt.imshow(mask)
plt.show()
np.count_nonzero(mask)

#In[]:

#In[]:
iann.getImgIds(iann.getAnnIds()[0: 10])
iann.loadAnns(iann.getAnnIds()[0: 2])

#In[]:
import numpy as np
print(iann.loadAnns.__doc__)
print(iann.getAnnIds.__doc__)
print(iann.getImgIds.__doc__)

print(len(iann.getAnnIds(catIds=[1], areaRng=[10000, np.inf], iscrowd=False)))
print(len(iann.getAnnIds(iscrowd=False)))
print(len(iann.getAnnIds()))
print(len(iann.getImgIds()))
print(len(set(iann.getAnnIds())))
all_annid = iann.getAnnIds()
print(iann.loadAnns(900100448263))
def bbox_area_stastics(catId):
    pass
#In[]:
class Statistic:
    def __init__(self):
        '''
        '''
        pass

    def config(self):
        '''
         @brief which indicator would be statistic
         @note
        '''
        # hw_rate
        # area
        # cat
        pass
    pass

def hw_rate():
    pass