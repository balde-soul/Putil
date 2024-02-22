##@FileName segmantic_to_bbox.py
# @Note 通过图像语义类型标注获取目标框类标注(VOC格式)，目前基于LV-MHP-v2编写
# @Author cjh
# @Time 2022-12-22
# coding=utf-8
import argparse, os, sys, cv2

parser = argparse.ArgumentParser()
parser.add_argument('--type_name', dest='TypeName', type=str, nargs='+', default=[], help='指定语义标记名称')
parser.add_argument('--label_var', dest='LabelVar', type=str, nargs='+', default=[], help='指定语义标记的值')
parser.add_argument('--target_type', dest='TargetType', type=str, nargs='+', default=[], help='指定目标语义')
parser.add_argument('--target_name', dest='TargetName', type=str, nargs='+', default=[], help='指定target_type映射为目标框类型')
parser.add_argument('--save_to', dest='SaveTo', type=str, default='', action='store', help='指定生成文件保存根目录')
options = parser.parse_args()
from ....path import touch_dir

print('generate ')
touch_dir(os.path.join(options.SaveTo, 'Annotations'))

# data loader
class MHP:
    def __init__(self, dataset_root, image_list_file=None):
        self._dataset_root = dataset_root
        self._image_root = os.path.join(self._dataset_root, 'images')
        self._anno_root = os.path.join(self.__dataset_root, 'annotations')
        if image_list_file is not None:
            with open(image_list_file, 'r') as fp:
                lines = fp.readlines()
        else:
            self._field = os.listdir(self._image_root)
            pass
        self._var_type = {
            0: 'background',
            1: 'hat',
            2: 'hair',
            3: 'sunglass',
            4: 'upper-clothes',
            5: 'skirt',
            6: 'pants',
            7: 'dress',
            8: 'belt ',
            9: 'left-shoe',
            10: 'right-shoe',
            11: 'face',
            12: 'left-leg',
            13: 'right-leg',
            14: 'left-arm ',
            15: 'right-arm',
            16: 'bag',
            17: 'scarf',
            18: 'torso-skin',
        }
        self._type_var = ~self._var_type
        pass

    @staticmethod
    def _img_to_anno(img_name):
        return '{0}.png'.format(img_name.split('.')[0])

    def __item__(self, index):
        img = cv2.imread(os.path.join(self._image_root, self._field[index]))
        label = cv2.imread(os.path.join(self._anno_root, MHP._img_to_anno(self._field[index])))
        return img, label
        pass

    def __len__(self):
        return len(self._field)
        pass
    pass


mhp_dataset = MHP()
for i in len(mhp_dataset):
    img, label = mhp_dataset[i]
    