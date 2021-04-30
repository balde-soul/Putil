# coding=utf-8
import numpy as np
import pandas as pd
import re
import os
import json
from enum import Enum
import Putil.data.common_data as pcd
import Putil.base.logger as plog

logger = plog.PutilLogConfig('voc').logger()
logger.setLevel(plog.DEBUG)

VOCLogger = logger.getChild('VOC')
VOCLogger.setLevel(plog.DEBUG)
VOC2012Logger = logger.getChild('VOC2012')
VOC2012Logger.setLevel(plog.DEBUG)

class VOC(pcd.CommonDataWithAug):
    binary_to_objname = {0: 'boat', 1: 'sheep', 2: 'bicycle', 3: 'aeroplane', 4: 'horse', 5: 'cow', 6: 'bottle', 7: 'train', 8: 'car',  \
        9: 'bus', 10: 'dog', 11: 'person', 12: 'chair', 13: 'cat', 14: 'bird', 15: 'sofa', \
            16: 'diningtable', 17: 'pottedplant', 18: 'motorbike', 19: 'tvmonitor', 20: 'object_type_amount'}
    objname_to_binary = {n: b for b, n in binary_to_objname.items()}
    binary_to_actname = {0: 'person_amount', 1: 'reading', 2: 'ridingbike', 3: 'jumping', 4: 'playinginstrument', 5: 'takingphoto', \
        6: 'walking', 7: 'ridinghorse', 8: 'usingcomputer', 9: 'running', 10: 'phoning', 11: 'action_type_amount'}
    actname_to_binary = {v: k for k, v in binary_to_actname.items()}
    class TaskType(Enum):
        ObjDet = 0
        Segmantation = 1
        Action = 2
        Layout = 3
        pass

    def __init__(
        self, 
        voc_root,
        task_type, 
        use_rate,
        sub_data,
        remain_strategy):
        pcd.CommonDataWithAug.__init__(self, use_rate=use_rate, sub_data=sub_data, remain_strategy=remain_strategy)
        self._voc_root = voc_root
        self._task_type = task_type
        if self._task_type == VOC.TaskType.ObjDet:
            pass
        pass

    @staticmethod
    def read_samples(fp):
        pass

    @staticmethod
    def _remove_nan(x):
        t = []
        for _x in x:
            if type(_x) == str:
                t += [_x]
                continue
            t += [] if np.isnan(_x) else [_x]
            pass
        return pd.Series(t)
    
    @staticmethod
    def _get_det_csv(root_dir, files, pattern, save_to, restatistic):
        df = None
        if os.path.exists(save_to) is False and restatistic is False:
            for _dsf in files:
                if _dsf in ['train.txt', 'val.txt', 'trainval.txt']:
                    continue
                if re.search(pattern, _dsf) is not None:
                    t = np.transpose(pd.read_csv(os.path.join(root_dir, _dsf), sep=' ', index_col=0, names=['object_type'] + list('adsas')).apply(VOC._remove_nan, axis=1)).rename({0: '{}'.format(_dsf.split('_')[0])})
                    df = t if df is None else pd.concat([df, t])
                    del t
                    pass
                pass
            df = df.replace({-1: 0., np.nan: 0})
            df = pd.concat([df, np.transpose(pd.DataFrame(df.sum())).rename({0: 'object_type_amount'})])
            object_amount_df = pd.DataFrame(df.sum(axis=1))
            object_amount_df.columns = ['total_smaple_base_image']
            object_amount_df.iloc[-1] = 0
            df = pd.concat([df, object_amount_df], axis=1)
            df.columns.name = None
            df.to_csv(save_to)
        else:
            df = pd.read_csv(save_to, sep=',', index_col=0)
        return df

    @staticmethod
    def _get_seg_csv(root_dir, file_name, save_to, restatistic):
        if os.path.exists(save_to) is False and restatistic is False:
            t = pd.concat([np.transpose(pd.read_csv(os.path.join(root_dir, file_name), sep=' ', index_col=None, names=['{}'.format(file_name.split('.')[0])] + list('dadskl')).apply(VOC._remove_nan, axis=1)), pd.DataFrame({0: [np.nan]})]).replace(np.nan, 1)
            new_header = t.iloc[0]
            t = t[1:]
            t.columns = new_header
            t = t.rename({0: file_name.split('.')[0]})
            t.columns.name = None
            t.to_csv(save_to)
        else:
            t = pd.read_csv(save_to, sep=',', index_col=0)
        return t

    @staticmethod
    def _get_act_csv(root_dir, files, pattern, save_to, restatistic):
        df = None
        if os.path.exists(save_to) is False and restatistic is False:
            for _dsf in files:
                VOCLogger.debug('processing action: {}'.format(_dsf))
                if _dsf in ['train.txt', 'val.txt', 'trainval.txt', '']:
                    continue
                if re.search(pattern, _dsf) is not None:
                    VOCLogger.debug('dealing {0} action.{1}'.format(pattern, _dsf))
                    t = pd.read_csv(os.path.join(root_dir, _dsf), sep=' ', index_col=0, names=['action_type'] + list('adsfh')).apply(VOC._remove_nan, axis=1)
                    t.columns = ['a', _dsf.split('_')[0]]
                    #.rename({0: 'person_amount'}).rename({1: '{}'.format(_dsf.split('_')[0])})
                    t.index.name = None
                    df = t.iloc[:, 0: 1][~t.index.duplicated()] if df is None else df
                    df.columns = ['person_amount'] if df.shape[1] == 1 else df.columns
                    df = pd.concat([df, t[~t.index.duplicated()].iloc[0: , 1:]], axis=1)
                    del t
                    pass
                pass
            df = df.replace({-1: 0., np.nan: 0.})
            action_amount = np.transpose(pd.DataFrame(df.iloc[:, :].sum())).rename({0: 'total_action_sample'})
            action_amount['person_amount'] = 0.
            df = pd.concat([df, action_amount])
            t = pd.DataFrame(df.sum(axis=1))
            t.columns = ['action_type_amount']
            t.iloc[-1] = 0.
            df = np.transpose(pd.concat([df, t], axis=1))
            df.to_csv(save_to)
        else:
            df = pd.read_csv(save_to, sep=',', index_col=0)
        return df

    ##@brief
    # @note 
    #   生成json的格式
    #   ordered_image_object：将iamge_id排序，使用map记录{image_id: index}
    #   action：在ordered_image_object的
    #   统计每个任务类型每个对象的样本量，交叉量
    # @param[in]
    # @param[in]
    # @return 
    @staticmethod
    def statistic(voc_root, save_to, restatistic=False):
        df = VOC.extract_data(voc_root, save_to, restatistic)
        pass

    @staticmethod
    def get_label(name, stage, task):
        return '{0}_{1}_{2}'.format(task.name, stage.name, name)

    @staticmethod
    def get_name(label):
        return label.split('_')[-1]

    @staticmethod
    def extract_data(voc_root, save_to, restatistic=False):
        statistic_dict = dict()
        images = os.listdir(os.path.join(voc_root, 'JPEGImages'))
        image_id_index_map = {image.split('.')[0]: index for index, image in enumerate(images)}
        statistic_dict['image_id_index_map'] = image_id_index_map

        ordered_image_object = [None] * len(image_id_index_map.keys())
        for k, v in image_id_index_map.items():
            ordered_image_object[v] = k
            pass
        statistic_dict['ordered_image_object'] = ordered_image_object

        _action_set_path = os.path.join(voc_root, 'ImageSets/Action')
        _action_set_files = os.listdir(_action_set_path)
        for _asf in _action_set_files:
            if re.search('trainval.txt', _asf) is not None:
                # todo: do the trainval
                continue
            if re.search('train.txt', _asf) is not None:
                # todo: do the train statistic
                continue
            if re.search('val.txt', _asf) is not None:
                # todo: do the val statistic
                continue
            pass

        _det_set_path = os.path.join(voc_root, 'ImageSets/Main')
        _det_set_files = os.listdir(_det_set_path)
        det_train_df_save_to = os.path.join(save_to, 'det_train.csv')
        det_val_df_save_to = os.path.join(save_to, 'det_val.csv')
        det_train_df = VOC._get_det_csv(_det_set_path, _det_set_files, '_train.txt', det_train_df_save_to, restatistic)
        det_train_df = det_train_df.rename({index: VOC.get_label(index, pcd.util.Stage.Train, VOC.TaskType.ObjDet) for index in  det_train_df.index})
        det_val_df = VOC._get_det_csv(_det_set_path, _det_set_files, '_val.txt', det_val_df_save_to, restatistic)
        det_val_df = det_val_df.rename({index: VOC.get_label(index, pcd.util.Stage.Evaluate, VOC.TaskType.ObjDet) for index in  det_val_df.index})
        det_df = pd.concat([det_train_df, det_val_df], axis=0)

        _seg_set_path = os.path.join(voc_root, 'ImageSets/Segmentation')
        seg_train_df_save_to = os.path.join(save_to, 'seg_train.csv')
        seg_val_df_save_to = os.path.join(save_to, 'seg_val.csv')
        seg_train_df = VOC._get_seg_csv(_seg_set_path, 'train.txt', seg_train_df_save_to, restatistic)
        seg_train_df = seg_train_df.rename({index: VOC.get_label(index, pcd.util.Stage.Train, VOC.TaskType.Segmantation) for index in  seg_train_df.index})
        seg_val_df = VOC._get_seg_csv(_seg_set_path, 'val.txt', seg_val_df_save_to, restatistic)
        seg_val_df = seg_val_df.rename({index: VOC.get_label(index, pcd.util.Stage.Evaluate, VOC.TaskType.Segmantation) for index in  seg_val_df.index})
        seg_df = pd.concat([seg_train_df, seg_val_df], axis=0)

        _act_set_path = os.path.join(voc_root, 'ImageSets/Action')
        _act_set_files = os.listdir(_act_set_path)
        act_train_df_save_to = os.path.join(save_to, 'act_train.csv')
        act_val_df_save_to = os.path.join(save_to, 'act_val.csv')
        act_train_df = VOC._get_act_csv(_act_set_path, _act_set_files, '_train.txt', act_train_df_save_to, restatistic)
        act_train_df = act_train_df.rename({index: VOC.get_label(index, pcd.util.Stage.Train, VOC.TaskType.Action) for index in  act_train_df.index})
        act_val_df = VOC._get_act_csv(_act_set_path, _act_set_files, '_val.txt', act_val_df_save_to, restatistic)
        act_val_df = act_val_df.rename({index: VOC.get_label(index, pcd.util.Stage.Evaluate, VOC.TaskType.Action) for index in  act_val_df.index})
        act_df = pd.concat([act_train_df, act_val_df], axis=0)

        df = pd.concat([det_df, seg_df, act_df], axis=0)
        df = df.replace(np.nan, 0.)
        return df

    @staticmethod
    def _get_sub_data(df, sub_data):
        t = list(set([index if VOC.get_name(d) in [VOC.binary_to_objname[sd] for sd in sub_data] else None for index, d in enumerate(df.index)]))
        t.remove(None)
        return df.iloc[t]

    @staticmethod
    def _get_df(df, stage, task):
        det_df = list(set([None if re.search('{1}_{0}'.format(stage.name, task.name), t) is None else index for index, t in enumerate(df.index)]))
        det_df.remove(None)
        det_df = df.iloc[det_df]
        return det_df

    @staticmethod
    def make_dataset(df, stage, det=False, action=False, seg=False, layout=False, \
        det_sub_data=None, action_sub_data=None, seg_sub_data=None, layout_sub_data=None, \
            task_set_operation=pcd.CommonData.TaskDataSetOperation.Cup):
        assert det or action or seg or layout is False

        det_df = VOC._get_df(df, stage, VOC.TaskType.ObjDet)
        det_set = VOC._get_sub_data(det_df, det_sub_data)

        seg_df = VOC._get_df(df, stage, VOC.TaskType.Segmantation)
        seg_set = VOC._get_sub_data(seg_df, seg_sub_data)
        pass

    @staticmethod
    def get_image_id(statistic_file):
        pass
    pass


##@brief
# @note
#   VOC2012详解：
#       统计量：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/dbstats.html
#       总共有20个类别id（不包括背景），
#       关键点：
#           **图像id**： 图像的唯一索引
#           **obj_name**：object的名称
#           **train**：训练数据，每个类别数据有5717个
#           **val**： 测试使用的数据集，每个类别数据有5832个
#           **trainval**：是**train**与**val**的并集
#           **
#       文件夹功能：
#           Annotations文件夹：存放着标记文件，以**图像id**做为名称的xml文件
#               标记格式：
#                   filename：图像名称
#                   object.name 目标名称
#                   object.bndbox 目标框，格式为（x_min,y_min,x_max,y_max）（Main标记）
#                   object.pose 目标姿态
#                   object.part 表示人物部件（有多个part）（Layout标记）
#                   object.actions 标记含有所有action的字段，用0表示不存在该action，用1表示存在该action（Action标记）
#                   segmented 表示该图像是否存在segmentation标记，存在的话可以去SegmentationClassAug或者SementationObject中查找标注
#                       **SegmentationClass是语义分割任务标记数据，像素语意标记，不区分同类不同个体**
#                       **SegmentationObject是实例分割任务标记数据，像素语意标记，区分同类不同个体**
#                       **SegmentationClassAug是个体轮廓标记数据，标记了每个目标的轮廓，不同目标使用相同的像素值标记**
#           JPEGImages文件夹：存放图像文件，以**图像id**作为名称的jpeg文件
#           ImageSets文件夹：顾名思义，这是一个存放图像集合的文件，至于集合边界条件，以下详解；
#               存放四个任务的对应数据，Action（动作）、Layout（人体部件语义）、Main（目标检测）、Segmentation（语义分割），
#               用于快速定位任务数据源（因为记录了对应任务存在标记的数据，以及数据的相关信息，有利于统计）
#               Action文件夹存放了关于人物任务动作分类的数据源，其中文件以**动作名**_<train/trainval/val>.txt的形式命名，格式为：每行一个**图像id**
#                   **图像id** **人物数量** **是否存在该**动作名****(1 -1 表示)，例如：2012_003420  4 -1表示的是图像2012_003420中有4个人，没有存在jumping动作 
#                   **集合边界：将包含任务相同动作的数据作为一个集合**
#               Layout文件夹存放人物人体部件的数据源，由于不存在多种，只需要记录哪些图像有人体部件标记以及有多少个人物，以train/trainval/val.txt命名，格式：
#                   每行一个**图像id**，一行为：**图像id** **人物数量**
#                   **集合边界：将包含人物部件标注的数据作为一个集合**
#               Main文件夹存放目标识别定位数据，总共有20个类别，其中文件以**obj_name**_train/trainval/val.txt;
#                   格式为：每行**图像id** **是否存在对应的obj（-1/1）**
#                   **集合边界：以存在相同obj的图像为一个集合**
#               Segmentation文件夹存放着图像语意分割几个，其中文件以train/trainval/val.txt的形式命名;
#                   **集合边界：将存在语意标注的数据作为一个集合**
#       标记详解：
#   
#
class VOC2012(VOC):
    ##@brief
    # @note
    # @param[in] voc_root, VOC的根目录，该目录包含Annotations，ImageSets等文件夹
    # @param[in] task_type, VOC20
    # @return 
    def __init__(
        self, 
        voc_root,
        task_type, 
        use_rate,
        sub_data,
        remain_strategy):
        VOC.__init__(self, voc_root, task_type, use_rate, sub_data, remain_strategy)
        pass
    pass