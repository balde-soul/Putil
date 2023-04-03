##@FileName VOCStatistic.py
# @Note 基于目标检测数据集VOC标记格式，提取统计数据，统计维度为[image_id, img_width, img_height, object_type1_bboxes, object_type${n}_bboxes, ..., all_object_type_name], \n
# object_type${n}_bbox以json编码:[[bbox_center_x/img_width, bbox_center_y/img_height, bbox_width/img_width, bbox_height/img_height, bbox_class, is_difficult]] \n
# 结果保存到xml_root的上层目录中，文件名为voc_xml_statistic.csv
# @Author cjh
# @Time 2022-12-08
# coding-utf-8
import pandas as pd
import xml.etree.ElementTree as ET
import os, copy, json, random, argparse, sys
random.seed(1995)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Putil.tools.data_process.voc_util import VOCToolSet

parser = argparse.ArgumentParser()
parser.add_argument('--xml_root', dest='XmlRoot', type=str, default='', help='指定xml存储路径')
parser.add_argument('--image_root', dest='ImageRoot', type=str, default='', help='指定图像集合根目录，如果不指定，可能出现xml文件纯在但image不存在的情况，\n这对统计是存在差错的')
parser.add_argument('--statistic_file', dest='StatisticFile', type=str, default='', help='指定统计结果保存位置(csv文件)，默认为空字符时，保存到xml_root上层目录/voc_xml_statistic.csv')
options = parser.parse_args()

if options.StatisticFile == '':
    result_name = 'voc_xml_statistic.csv'
    os.path.join(os.path.dirname(options.XmlRoot), result_name)
    pass
else:
    if not os.path.exists(os.path.dirname(options.StatisticFile)):
        print('statistic_file: {0}, root dir not found'.format(options.StatisticFile))
        sys.exit(1)
    pass

if options.ImageRoot == '':
    print('可能出现xml文件纯在但image不存在的情况，这容易出现统计出错，直接使用xml路径进行统计')
    xmls = os.listdir(options.XmlRoot)
    pass
else:
    xmls = list()
    for img in os.listdir(options.ImageRoot):
        imgdir = os.path.join(options.ImageRoot, img)
        if os.path.exists(os.path.join(options.XmlRoot, VOCToolSet.id2xml(VOCToolSet.image2id(imgdir)))):
            xmls.append(VOCToolSet.id2xml(VOCToolSet.image2id(imgdir)))

def convert(size, box):
    dw = 1./(size[0])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dw = 1./((size[0])+0.1)
    dh = 1./(size[1])  # 有的人运行这个脚本可能报错，说不能除以0什么的，你可以变成dh = 1./((size[0])+0.1)
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

statistic = list()

for xml in xmls:
    in_file = open(os.path.join(options.XmlRoot, xml), encoding='utf-8')
    tree = ET.parse(in_file)
    root = tree.getroot()
    image_name = root.find('filename').text
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    target_bbox = dict()
 
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        object_type = obj.find('name').text
        target_bbox[object_type] = list() if object_type not in target_bbox.keys() else target_bbox[object_type]
        # 非目标类别以及标记为困难标注类型的被忽略
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        target_bbox[object_type].append(list(bb) + [object_type, int(difficult)])
    se = {ot: json.dumps(bbox) for ot, bbox in target_bbox.items()}
    for ot, bbox in target_bbox.items():
        se['{0}_object_amount'.format(ot)] = len(bbox)
    se['image_id'] = image_name
    se['image_width'] = w
    se['image_height'] = h
    se['type_set'] = json.dumps([ot for ot, bbox in target_bbox.items()])
    statistic.append(se)
    pass
df = pd.DataFrame(statistic)

type_set = list()
def get_all_type(x):
    type_set_list = json.loads(x['type_set'])
    for ts in type_set_list:
        if ts not in type_set:
            type_set.append(ts)
            pass
        pass
    pass
df.apply(get_all_type, axis=1)
type_object_amount = {ts: df['{0}_object_amount'.format(ts)].sum() for ts in sorted(type_set)}
print('all types: {0}'.format(type_object_amount))

def bbox_statistic(x):
    type_set_list = json.loads(x['type_set'])
    ret = dict()
    for ts in type_set_list:
        return json.loads(x['ts'])
        pass
    pass

print(df.to_csv(options.StatisticFile), sep=',')