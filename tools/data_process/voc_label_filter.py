# coding=utf-8
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

parser = argparse.ArgumentParser()
parser.add_argument('--xml_root', dest='XmlRoot', type=str, default='', action='store', help='指定xml根目录')
parser.add_argument('--save_to', dest='SaveTo', type=str, default='', action='store', help='指定结果保存路径')
parser.add_argument('--in', dest='In', type=str, default=[], nargs='+', help='指定需要保留的类集合')
parser.add_argument('--out', dest='Out', type=str, default=[], nargs='+', help='指定需要过滤除去的类集合')
options = parser.parse_args()
from Putil.path import touch_dir
from Putil.tools.data_process.voc_util import VOCToolSet

if len(options.In) * len(options.Out) != 0:
    print('specified in or out, do not specified both: in: {0}, out: {1}'.format(options.In, options.Out))
    sys.exit(1)
    pass

if len(options.In) == 0 and len(options.Out) == 0:
    print('specified in or out, both in and out is not spcified')
    sys.exit(1)
    pass

if len(options.In) != 0:
    def filter(name):
        if name in options.In:
            return True
        else:
            return False
        pass
    pass
else:
    def filter(name):
        if name in options.Out:
            return False
        else:
            return True
            pass
        pass
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
        if filter(o['name']):
            remain_objects.append(o)
        else:
            pass
        pass
    VOCToolSet.append_object(nxml, remain_objects)
    pass