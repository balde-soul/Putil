# coding=utf-8
import re
import os
import json
from enum import Enum
import Putil.data.common_data as pcd
import Putil.base.logger as plog

logger = plog.PutilLogConfig('voc').logger()
logger.setLevel(plog.DEBUG)

VOC2012Logger = logger.getChild('VOC2012')
VOC2012Logger.setLevel(plog.DEBUG)

class VOC(pcd.CommonDataWithAug):
    class TaskType(Enum):
        ObjDet = 0
        Segmantation = 1
        Action = 2
        Layout = 3
        pass

    ##@brief
    # @note 
    #   生成json的格式
    #   ordered_image_object：将iamge_id排序，使用map记录{image_id: index}
    #   action：在ordered_image_object的
    # @param[in]
    # @param[in]
    # @return 
    @staticmethod
    def statistic(voc_root, sub_data, fp):
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
                with open(_asf, 'r') as fp:
                    pass
                continue
            if re.search('val.txt', _asf) is not None:
                # todo: do the val statistic
                continue
        pass

    @staticmethod
    def get_image_id(statistic_file, ):
        pass
    pass


##@brief
# @note
#   VOC2012详解：
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
#                       **SegmentationClass是语义分割任务标记数据，标记提供了轮廓，需要生成语义**
#                       **SegmentationObject是实例分割任务标记数据**
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
        pcd.CommonDataWithAug.__init__(self, use_rate=use_rate, sub_data=sub_data, remain_strategy=remain_strategy)
        pass
    pass