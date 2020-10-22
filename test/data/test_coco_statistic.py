#In[]
import argparse
options = argparse.ArgumentParser()
options.add_argument('--test_img_amount', action='store_true', default=False)
options.add_argument('--test_obj_size_follow_cat', action='store_true', default=False)
args = options.parse_args()
# coding=utf-8
#class a:
#    a = 1
#    b = 2
#    c = 1
#    d = 1
ann_file = '/data2/Public_Data/COCO/unzip_data/2017/annotations/instances_train2017.json'
save_to_path = './test/data/result/coco_statistic'

import os
import Putil.data.coco as coco

if os.path.exists(save_to_path) is False:
    os.mkdir(save_to_path)

#In[]
if args.test_obj_size_follow_cat:
    #for cat_id, cat_name in coco.COCOBase._detection_cat_id_to_cat_name.items():
    cat_names = list(coco.COCOBase._detection_cat_name_to_represent.keys())
    coco.COCOBase.detection_statistic_obj_size_follow_cat(cat_names, ann_file, save_to_path)

#In[]
if args.test_img_amount:
    coco.COCOBase.detection_statistic_img_amount(ann_file=ann_file, save_to=save_to_path)