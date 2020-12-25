#In[]
import os
import argparse
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
options = argparse.ArgumentParser()
options.add_argument('--test_img_amount', action='store_true', default=False)
options.add_argument('--test_obj_size_follow_cat', action='store_true', default=False)
options.add_argument('--test_coco_basical_statistic', action='store_true', default=False)
args = options.parse_args()
# coding=utf-8
#class a:
#    a = 1
#    b = 2
#    c = 1
#    d = 1
coco_root_dir = '/data2/Public_Data/COCO/unzip_data/2017'
train_ann_file = os.path.join(coco_root_dir, 'annotations/instances_train2017.json')
evaluate_ann_file = os.path.join(coco_root_dir, 'annotations/instances_val2017.json')
save_to_path = './test/data/result/coco_statistic'

import Putil.data.coco as coco

if os.path.exists(save_to_path) is False:
    os.mkdir(save_to_path)
train_save_to_path = os.path.join(save_to_path, 'train')
if os.path.exists(train_save_to_path) is False:
    os.mkdir(train_save_to_path)
evaluate_save_to_path = os.path.join(save_to_path, 'evaluate')
if os.path.exists(evaluate_save_to_path) is False:
    os.mkdir(evaluate_save_to_path)

#In[]
if args.test_coco_basical_statistic:
    coco.COCOBase.coco_basical_statistic(coco_root_dir, save_to_path)
#In[]
if args.test_obj_size_follow_cat:
    cat_names = list(coco.COCOBase._detection_cat_name_to_represent.keys())
    coco.COCOBase.detection_statistic_obj_size_follow_cat(cat_names=cat_names, ann_file=train_ann_file, save_to=train_save_to_path)
    coco.COCOBase.detection_statistic_obj_size_follow_cat(cat_names=cat_names, ann_file=evaluate_ann_file, save_to=evaluate_save_to_path)

#In[]
if args.test_img_amount:
    coco.COCOBase.detection_statistic_img_amount_obj_amount(ann_file=train_ann_file, save_to=train_save_to_path)
    coco.COCOBase.detection_statistic_img_amount_obj_amount(ann_file=evaluate_ann_file, save_to=evaluate_save_to_path)

#In[]
#import pandas as df
#import numpy as np
#import matplotlib.pyplot as plt
#
#data = np.random.sample(10000000)
#
#plt.rcParams['savefig.dpi'] = 300
#a = df.DataFrame({'a': data.tolist()})
#a.plot.hist(grid=True, bins=500, rwidth=0.9, color='#607c8e')
#plt.title('test1')
#plt.ylabel('Counts')
#plt.xlabel('bbox area/100')
#plt.savefig('./test1.png')
#
#plt.rcParams['savefig.dpi'] = 300
#a = df.DataFrame({'b': (data * 9).tolist()})
#a.plot.hist(grid=True, bins=500, rwidth=0.9, color='#607c8e')
#plt.title('test2')
#plt.ylabel('Counts')
#plt.xlabel('bbox area/100')
#plt.savefig('./test2.png')