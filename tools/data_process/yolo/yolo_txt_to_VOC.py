##@FileName yolo_txt_to_VOC.py
# @Note 将yolo的txt标记格式转为VOC
# @Author cjh
# @Time 2022-12-09
# coding=utf-8

# 若模型保存文件夹不存在，创建模型保存文件夹，若存在，删除重建
from glob import glob
import cv2, copy, json, random, sys, cv2, argparse, os
from lxml.etree import Element, SubElement, tostring
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))
import pandas as pd
random.seed(1995)

parser = argparse.ArgumentParser()
parser.add_argument('--image_list_file_path', dest='ImageListFilePath', type=str, default='', action='store', help='表示图像列表的txt文件或者图像集路径')
parser.add_argument('--txt_root', dest='TxtRoot', type=str, action='store', default='', help='保存txt文件的路径')
parser.add_argument('--xml_root', dest='XmlRoot', type=str, action='store', default='', help='保存xml文件的路径')
parser.add_argument('--category_set', dest='CategorySet', type=str, nargs='+', default=[], help='会根据category_set的内容配合txt中的object index生成目标名称')
options = parser.parse_args()
from Putil.tools.data_process.voc_util import VOCToolSet


# YOLO格式的txt转VOC格式的xml
def convert(img, box):
    name, x, y, w, h = box
    img_w = img.shape[0]
    img_h = img.shape[1]
    x = float(x) * img_h
    w = float(w) * img_h
    y = float(y) * img_w
    h = float(h) * img_w
    x = (x * 2 - w) / 2
    y = (y * 2 - h) / 2
    return name, x, y, w, h

# 单个文件转换
def txt_xml(img_path, txt_path, xml_save_to, category_set):
    clas = []
    img_name = os.path.split(img_path)[-1]
    img_id = img_name.split('.')[0]
    img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    imh, imw = img.shape[0: 2]
    txt_img = txt_path
    with open(txt_img, "r", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip('\n')
            list = line.split(" ")
            list = convert(img, list)
            clas.append(list)
    node_root = Element('annotation')
    node_folder = SubElement(node_root, 'folder')
    node_folder.text = 'imge'
    node_filename = SubElement(node_root, 'filename')
    node_filename.text = img_name
    node_filepath = SubElement(node_root, 'path')
    node_filepath.text = img_path
    node_size = SubElement(node_root, 'size')
    node_width = SubElement(node_size, 'width')
    node_width.text = str(imw)
    node_height = SubElement(node_size, 'height')
    node_height.text = str(imh)
    node_depth = SubElement(node_size, 'depth')
    node_depth.text = '3'
    for i in range(len(clas)):
        node_object = SubElement(node_root, 'object')
        node_name = SubElement(node_object, 'name')
        node_name.text = str(category_set[int(clas[i][0])])
        node_pose = SubElement(node_object, 'pose')
        node_pose.text = "Unspecified"
        node_truncated = SubElement(node_object, 'truncated')
        node_truncated.text = '0'
        node_difficult = SubElement(node_object, 'difficult')
        node_difficult.text = '0'
        node_bndbox = SubElement(node_object, 'bndbox')
        node_xmin = SubElement(node_bndbox, 'xmin')
        node_xmin.text = str(int(clas[i][1] + 0.5))
        node_ymin = SubElement(node_bndbox, 'ymin')
        node_ymin.text = str(int(clas[i][2] + 0.5))
        node_xmax = SubElement(node_bndbox, 'xmax')
        node_xmax.text = str(int(clas[i][1] + clas[i][3] + 0.5))
        node_ymax = SubElement(node_bndbox, 'ymax')
        node_ymax.text = str(int(clas[i][2] + clas[i][4] + 0.5))
    xml = tostring(node_root, pretty_print=True)  # 格式化显示，该换行的换行
    img_newxml = os.path.join(xml_save_to, img_id + '.xml')
    file_object = open(img_newxml, 'wb', encoding='utf-8')
    file_object.write(xml)
    file_object.close()


# 批量转换
def generate_label_file():
    # 获取所有图片
    imgs = []
    jpg_imgs = glob("{}/*.jpg".format(img_path))
    png_imgs = glob("{}/*.png".format(img_path))
    imgs.extend(jpg_imgs)
    imgs.extend(png_imgs)
    for img in imgs:
        txt_xml(img.split(os.sep)[-1])

if options.XmlRoot == '':
    print('specify the xml_root')
    sys.exit(1)
    pass
if not os.path.exists(options.XmlRoot):
    print('mkdir {0}'.format(options.XmlRoot))
    os.mkdir(options.XmlRoot)
    pass

if options.ImageListFilePath == '':
    print('specify the image_list_file_path')
    sys.exit(1)
    pass

if options.TxtRoot == '':
    print('specify the txt_root')
    pass

if len(options.CategorySet) == 0:
    print('specify the category_set')
    pass

if os.path.isdir(options.ImageListFilePath):
    img_paths = [os.path.join(options.ImageListFilePath, i) for i in os.listdir(options.ImageListFilePath)]
elif os.path.isfile(options.ImageListFilePath):
    with open(options.ImageListFilePath, 'r') as fp:
        img_paths = [i.strip('\n') for i in fp.readlines()]
else:
    print('image_list_file_path: {0} type no supported'.format(options.ImageListFilePath))
    sys.exit(1)
    pass
with open(os.path.join(os.path.dirname(os.path.abspath(options.XmlRoot)), '{0}.info'.format(os.path.split(os.path.abspath(__file__))[1].split('.')[0])), 'w', encoding='utf-8') as info_fp:
    for img_path in img_paths:
        if img_path is None:
            break
        img_id = VOCToolSet.image2id(img_path)
        txt_path = os.path.join(options.TxtRoot, '{0}.txt'.format(img_id))
        if not os.path.exists(img_path):
            info_fp.write('{0} does not exist\n'.format(img_path))
            continue
        if not os.path.exists(txt_path):
            info_fp.write('{0} does not exist\n'.format(txt_path))
            continue

        clas = []
        img_name = os.path.split(img_path)[-1]
        img_id = img_name.split('.')[0]
        img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        imh, imw = img.shape[0: 2]
        txt_img = txt_path
        with open(txt_img, "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                _list = line.split(" ")
                _list = convert(img, _list)
                clas.append(_list)
        objects = list()
#     * [{'name': str, 'bbox': {'xmin': int, 'ymin': int, 'xmax': int, 'ymax': int}, 'pose': str, 'truncated': int, 'difficult': int}]
        for i in range(len(clas)):
            objects.append(
                {
                    'name': str(options.CategorySet[int(clas[i][0])]),
                    'bbox': {
                        'xmin': int(clas[i][1] + 0.5),
                        'ymin': int(clas[i][2] + 0.5),
                        'xmax': int(clas[i][1] + clas[i][3] + 0.5),
                        'ymax': int(clas[i][2] + clas[i][4] + 0.5)
                    }
                }
            )
        VOCToolSet.generate_empty_xml(img_path, os.path.join(options.XmlRoot, VOCToolSet.id2xml(img_id)))
        VOCToolSet.append_object(os.path.join(options.XmlRoot, VOCToolSet.id2xml(img_id)), objects)
        #txt_xml(img_path, txt_path, options.XmlRoot, options.CategorySet)
        pass
pass