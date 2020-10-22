# coding=utf-8
#class a:
#    a = 1
#    b = 2
#    c = 1
#    d = 1
ann_file = '/data2/Public_Data/COCO/unzip_data/2017/annotations/instances_val2017.json'
save_to_path = './test/data/result/coco_statistic'

import os
import Putil.data.coco as coco

if os.path.exists(save_to_path) is False:
    os.mkdir(save_to_path)

#for cat_id, cat_name in coco.COCOBase._detection_cat_id_to_cat_name.items():
cat_names = list(coco.COCOBase._detection_cat_name_to_represent.keys())
coco.COCOBase.detection_statistic_obj_size_follow_cat(cat_names, ann_file, save_to_path)