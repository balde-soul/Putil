# coding=utf-8

from pycocotools.coco import COCO
from optparse import OptionParser
parser = OptionParser(usage='usage %prog [options] arg1 arg2')
level_default = 'Debug'
parser.add_option(
    '--level',
    action='store',
    dest='Level',
    type=str,
    default=level_default,
    help='specify the log level for the app'
         'default: {0}'.format(level_default)
)
coco_dir_default = ''
parser.add_option(
    '--coco_dir',
    action='store',
    dest='CocoDir',
    type=str,
    default=coco_dir_default,
    help='specify the direction of the coco data set annotations'
         'default: {0}'.format(coco_dir_default)
)
ann_file_name_default = ''
parser.add_option(
    '--ann_file_name',
    action='store',
    dest='AnnFileName',
    type=str,
    default=ann_file_name_default,
    help='specify the annotation file name list which would be calc'
         'sefault: {0}'.format(ann_file_name_default)
)
calc_default = 'all'
parser.add_option(
    '--calc',
    action='store',
    dest='Calc',
    type=str,
    default=calc_default,
    help='specify which kind of indicator to calculation'
         'default: {0}'.format(calc_default)
)
store_dir = ''
parser.add_option(
    '--store_dir',
    action='store',
    dest='StoreDir',
    type=str,
    default=store_dir,
    help='specify the path which used to store the indicator'
         'default: {0}'.format('')
)

import Putil.loger as plog
import os
from colorama import Fore
import matplotlib.pyplot as plt
(options, args) = parser.parse_args()
plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
plog.PutilLogConfig.config_handler(plog.stream_method)
plog.PutilLogConfig.config_format(
    "%(filename)s: %(lineno)d: %(levelname)s: %(name)s: %(message)s")

root_logger = plog.PutilLogConfig('COCO/basal_calc').logger()
root_logger.setLevel(plog.DEBUG)

CocoIndicatorLogger = root_logger.getChild('CocoIndicator').logger()
CocoIndicatorLogger.setLevel(plog.DEBUG)

coco = None

'''
    declare:
        this class is is base on coco2017
        calculation some indicator of the box:
            the plot of box-width vs. box-height
            the histogram of box-count in signal image
            the plot of box-y-shift-center_y vs. box-x-shift-center_x of all data
            the histogram of class-obj-count in all data
'''


class CocoIndicator:
    def __init__(self, ann_file, store_dir, **options):
        self._coco = COCO(ann_file)
        self._store_dir = store_dir
        self._image_id = self._coco.getImgIds()

        self._box_h_w_collection = list()
        self._signal_image_object_count_collection = dict()
        self._box_y_x_shift_collection = list()
        self._obj_class_count_collection = dict()
        self._image_h_w_collection = list()
        self._box_y_x_shift_normal_collection = list()

        self.BHW = options.pop('BHW', False)
        CocoIndicatorLogger.info(
            Fore.LIGHTGREEN_EX + 'BHW: generate box_h_w_distribution: {0}'.format(self.BHW) + Fore.RESET)
        self.SIOC = options.pop('SIOC', False)
        CocoIndicatorLogger.info(
            Fore.LIGHTGREEN_EX + 'SIOC: generate signal_image_object_count_histogram: {0}'.format(
                self.SIOC) + Fore.RESET)
        self.BYXS = options.pop('BYXS', False)
        CocoIndicatorLogger.info(
            Fore.LIGHTGREEN_EX + 'BYXS: generate box_y_x_shift_distribution: {0}'.format(
                self.BYXS) + Fore.RESET)
        self.OCC = options.pop('OCC', False)
        CocoIndicatorLogger.info(
            Fore.LIGHTGREEN_EX + 'OCC: generate obj_class_count_histogram: {0}'.format(
                self.OCC) + Fore.RESET)
        self.IHW = options.pop('IHW', False)
        CocoIndicatorLogger.info(
            Fore.LIGHTGREEN_EX + 'IHW: generate image_h_w_distribution: {0}'.format(
                self.IHW) + Fore.RESET)
        self.BYXSN = options.pop('BYXSN', False)
        CocoIndicatorLogger.info(
            Fore.LIGHTGREEN_EX + 'SIOC: generate box_y_x_shift_normal_distribution: {0}'.format(
                self.BYXSN) + Fore.RESET)
        pass

    def Calc(self):
        for imgId in self._image_id:
            image = self._coco.loadImgs(imgId)
            image_height = image['height']
            image_width = image['width']
            image_id = image['id']
            self.__update_image_h_w_collection(image_height, image_width) if self.IHW else None
            annIds = self._coco.getAnnIds(imgIds=image_id)
            self.__update_signal_image_obj_count(len(annIds)) if self.SIOC else None
            for annId in annIds:
                ann = self._coco.loadAnns(annId)
                box = ann['box']
                box_height = box[3]
                box_width = box[2]
                box_center_y = box[1]
                box_center_x = box[0]
                classify = ann['category_id']
                self.__update_box_h_w_distribution(box_height, box_width) if self.BHW else None
                self.__update_box_y_x_shift(image_height, image_width, box_center_y, box_center_x) if self.BYXS else None
                self.__update_box_y_x_shift_normal(image_height, image_width, box_center_y, box_center_x) if self.BYXSN else None
                self.__update_obj_class_count(classify) if self.OCC else None
                pass
            pass

        CocoIndicatorLogger.info(Fore.LIGHTGREEN_EX + 'generate visual at: {0}'.format(self._store_dir) + Fore.RESET)

        pass

    def __box_h_w_distribution_visual(self):
        plt.ax
        pass

    def __update_box_h_w_distribution(self, box_height, box_width):
        self._box_h_w_collection.append([box_height, box_width])
        pass

    def __update_signal_image_obj_count(self, obj_amount):
        if obj_amount in self._signal_image_object_count_collection.keys():
            self._signal_image_object_count_collection[obj_amount] += 1
        else:
            self._signal_image_object_count_collection[obj_amount] = 1
        pass

    def __update_box_y_x_shift(self, image_height, image_width, box_center_y, box_center_x):
        image_center_y = image_height / 2.0
        image_center_x = image_width / 2.0
        self._box_y_x_shift_collection.append(
            [box_center_y - image_center_y, box_center_x - image_center_x]
        )
        pass

    def __update_obj_class_count(self, classify):
        if classify in self._obj_class_count_collection.keys():
            self._obj_class_count_collection[classify] += 1
        else:
            self._obj_class_count_collection[classify] = 1
        pass

    def __update_image_h_w_collection(self, image_height, image_width):
        self._image_h_w_collection.append([image_height, image_width])
        pass

    def __update_box_y_x_shift_normal(self, image_height, image_width, box_center_y, box_center_x):
        image_center_y = image_height / 2.0
        image_center_x = image_width / 2.0
        self._box_y_x_shift_normal_collection.append(
            [box_center_y - image_center_y / image_center_y,
             box_center_x - image_center_x / image_center_x])
        pass


class CcoColass:
    def __init__(self, ann_file, store_file):
        self._coco = COCO(ann_file)
        self._store_dir = store_dir
        pass


if __name__ == '__main__':
    ann_file = os.path.join(options.CocoDir, options.AnnFileName)
    if options.StoreDir == '':
        store_dir_box = os.path.join(options.CocoDir, 'box')
        pass
    else:
        store_dir_box = os.path.join(options.StoreDir, 'box')
        pass
    if options.Calc == 'all':
        CocoBox(ann_file, store_dir_box)
        pass
    pass

#In[]:
import torchvision.datasets as dset
path2data="/data2/Public_Data/COCO/train2017"
path2json="/data2/Public_Data/COCO/annotations/instances_train2017.json"
coco_train = dset.CocoDetection(root = path2data, annFile = path2json)