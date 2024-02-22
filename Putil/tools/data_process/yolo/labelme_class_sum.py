import xml.etree.ElementTree as ET
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--xml_root', dest='XmlRoot', type=str, default='', help='指定xml存储路径')
parser.add_argument('--output_file', dest='OutputFile', type=str, default='', help='指定输出文件路径')
options = parser.parse_args()

if not os.path.exists(options.XmlRoot):
    sys.exit(1)
    pass

if options.OutputFile == '':
    options.OutputFile = os.path.dirname(options.XmlRoot)
    options.OutputFile = os.path.join(options.OutputFile, '{0}.{1}'.format(os.path.split(__file__)[-1].split('.')[0], 'yaml'))
    pass

object_types = list()
try:
    for xml_file in os.listdir(options.XmlRoot):
        in_file = os.path.join(options.XmlRoot, xml_file)
        tree = ET.parse(in_file)
        root = tree.getroot()
        image_name = root.find('filename').text
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        object_one_hot = dict()
        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            object_type = obj.find('name').text
            if object_type not in object_types:
                object_types.append(object_type)
            pass
        pass
except Exception as ex:
    print('process file: {0} exception occured'.format(in_file))
    raise ex
print(object_types)