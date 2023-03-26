# coding=utf-8
from msilib.schema import Error
import os, sys, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import argparse
from Putil.tools.data_process.voc_util import VOCToolSet

parser = argparse.ArgumentParser(
    '本工具用于分离voc数据集，基本操作是根据xml或者image提取原始集合中的image或xml'
)
parser.add_argument('--o_set_image_root', dest='OSetImageRoot', type=str, action='store', default='', help='指定原始集合图像根路径')
parser.add_argument('--o_set_xml_root', dest='OSetXmlRoot', type=str, action='store', default='', help='指定原始集合标记根路径')
parser.add_argument('--t_set_image_root', dest='TSetImageRoot', type=str, action='store', default='', help='指定分离出来的集合图像根路径')
parser.add_argument('--t_set_xml_root', dest='TSetXmlRoot', type=str, action='store', default='', help='指定分离出来的集合标记根路径')
parser.add_argument('--xml_mode', dest='XmlMode', action='store_true', default=False, help='使用xml模式，即根据TSetXmlRoot中文件从OSetImageRoot中提取图像到TSetImageRoot')
parser.add_argument('--image_mode', dest='ImageMode', action='store_true', default=False, help='使用image模式，即根据TSetImageRoot中文件从OSetXmlRoot中提取标记到TSetXmlRoot')
options = parser.parse_args()

ImageRoot = r'D:\download_from_server\MyDataset\JPEGImages'

XmlRoot = r'D:\download_from_server\SCUT-Filtered\a'
ToImageRoot = r'D:\download_from_server\SCUT-Filtered\JPEGImages-ForChefHat-Manualed'

if not (options.ImageMode or options.XmlMode):
    print('no mode specified')
    sys.exit(1)

if options.XmlMode:
    xmls = [os.path.join(options.TSetXmlRoot, xml) for xml in os.listdir(options.TSetXmlRoot)]
    img2img = [(os.path.join(options.OSetImageRoot, VOCToolSet.get_image_info(xml)['filename']), options.TSetImageRoot) for xml in xmls]
else:
    img2img = []
    pass


if options.ImageMode:
    images = [os.path.join(options.TSetImageRoot, image) for image in os.listdir(options.TSetImageRoot)]
    xml2xml = [(os.path.join(options.OSetXmlRoot, VOCToolSet.image2id(image)), options.TSetXmlRoot) for image in images]
else:
    xml2xml = []

operations = img2img + xml2xml
#print(operations)
#operations = []

succ = 0
for op in operations:
    try:
        shutil.move(op[0], op[1])
        succ += 1
    except FileNotFoundError as ex:
        print(ex)
    except shutil.Error as ex:
        print(ex)

    #python .\tools\data_process\voc_seperator.py --o_set_xml_root E:\download_from_server\20230324\AnnotationsFromYolo-Reflect-ForChefHat-Auto --o_set_image_root E:\download_from_server\20230324\JPEGImages --t_set_xml_root E:\download_from_server\20230324\out-ann --t_set_image_root E:\download_from_server\20230324\out-img\ --xml_mode --image_mode