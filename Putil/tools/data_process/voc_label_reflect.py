# coding=utf-8
##@FileName voc_label_reflect.py
# @Note 本脚本用于变换voc标记文件中的类别名
# @Author cjh
# @Time 2023-03-14
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument('--xml_root', dest='XmlRoot', type=str, default='', action='store', help='指定xml根目录')
parser.add_argument('--save_to', dest='SaveTo', type=str, default='', action='store', help='指定结果保存路径')
parser.add_argument('--original_category', dest='OriginalCategory', type=str, nargs='+', default=[], help='指定待映射原始label名称集合')
parser.add_argument('--target_category', dest='TargetCategory', type=str, nargs='+', default=[], help='指定映射之后的label名称，需要与original_category同等长度')
options = parser.parse_args()
from Putil.path import touch_dir
from Putil.tools.data_process.voc_util import VOCToolSet

if len(options.OriginalCategory) != len(options.TargetCategory):
    print('reflect map: not complete {0}(from)->{1}(to)'.format(options.OriginalCategory, options.TargetCategory))
    sys.exit(1)
    pass
else:
    reflect_map = {o: t for o, t in zip(options.OriginalCategory, options.TargetCategory)}
    print(reflect_map)
    pass


if not os.path.exists(options.XmlRoot):
    print('xml_root: {0} does not exist'.format(options.XmlRoot))
    sys.exit(1)
    pass

if not os.path.exists(options.SaveTo):
    touch_dir(options.SaveTo)
    pass

xmls = [os.path.join(options.XmlRoot, xr) for xr in os.listdir(options.XmlRoot)]
for xml in xmls:
    fid = VOCToolSet.xml2id(xml)
    nxml = os.path.join(options.SaveTo, VOCToolSet.id2xml(fid))
    objects = VOCToolSet.extract_object(xml)
    image_info = VOCToolSet.extract_image_info(xml)
    VOCToolSet.generate_empty_xml(image_info, nxml)
    remain_objects = list()
    for o in objects:
        o['name'] = reflect_map.get(o['name'], o['name'])
        remain_objects.append(o)
        pass
    VOCToolSet.append_object(nxml, remain_objects)
    pass