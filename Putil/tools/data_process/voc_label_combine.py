##@FileName voc_label_combine.py
# @Note 本文件用于合并VOC标记，基于单个JPEGImages目录，将多个Annotations目录的标记合并，将合并的标记文件保存到Combine_save_to
# @Author cjh
# @Time 2023-01-04
# coding=utf-8
import argparse, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
parser = argparse.ArgumentParser()
parser.add_argument('--image_root', dest='ImageRoot', type=str, default='', action='store', help='指定图像根目录')
parser.add_argument('--xml_roots', dest='XmlRoots', type=str, default=[], nargs='+', help='指定希望进行合并的标记路径集合')
parser.add_argument('--combine_save_to', dest='CombineSaveTo', type=str, default='', action='store', help='合并的标记保存路径')
parser.add_argument('--ious', dest='IoUS', action='store_true', help='是否使用IoU进行标记过滤,当指定--ious时，xml_roots中的标记如果IoU超过--ious_thres，则丢弃其中一个，xml_roots中前置位优先')
parser.add_argument('--ious_t', dest='IoUST', type=float, default=0.7, action='store', help='当使用--ious时需要指定的参数, 默认为0.9')
#parser.add_argument('--operation', dest='')
options = parser.parse_args()

# <block_begin: 测试使用
# @time 2023-01-04
# @author cjh
#options.ImageRoot = '/data/caojihua/data/1229-fight-run-jump/run/JPEGImages/'
#options.XmlRoots = ['/data/caojihua/data/1229-fight-run-jump/run/Annotations/', '/data/caojihua/data/1229-fight-run-jump/run/AnnotationsFromYolov5v70Large/']
#options.CombineSaveTo = '/data/caojihua/data/1229-fight-run-jump/run/AnnotationsCombineYoloPreAndRunLabel/'
# block_end: >

from Putil.path import touch_dir
from Putil.tools.data_process.voc_util import VOCToolSet
from Putil.calc.iou import calc_iou_matrix_thw

if options.ImageRoot == '':
    print('no image_root specified')
    sys.exit(1)
else:
    if not os.path.exists(options.ImageRoot):
        print('image_root {0} is not existed'.format(options.ImageRoot))
        sys.exit(1)
        pass
    pass

if options.CombineSaveTo == '':
    print('no combine_save_to specified')
    sys.exit(1)
else:
    if not os.path.exists(options.CombineSaveTo):
        touch_dir(options.CombineSaveTo)
        pass
    print('save to {0}'.format(options.CombineSaveTo))
    pass

if len(options.XmlRoots) == 0:
    print('no target specified')
    sys.exit(1)
    pass
else:
    for xr in options.XmlRoots:
        if not os.path.exists(xr):
            print('xml_root: {0} is not existed or not file'.format(xr))
            sys.exit(1)
            pass
        pass
    pass

imgdir_set = [os.path.join(options.ImageRoot, ir) for ir in os.listdir(options.ImageRoot)]
label_set = {xr: [VOCToolSet.xml2id(os.path.join(xr, xml)) for xml in os.listdir(xr)] for xr in options.XmlRoots}
for imgdir in imgdir_set:
    fid = VOCToolSet.image2id(imgdir)
    new_xml_dir = os.path.join(options.CombineSaveTo, VOCToolSet.id2xml(fid))
    VOCToolSet.generate_empty_xml(imgdir, new_xml_dir)
    for rt, ls in label_set.items():
        if fid in ls:
            ocs_old = VOCToolSet.extract_object(new_xml_dir)
            VOCToolSet.generate_empty_xml(imgdir, new_xml_dir)
            ocs = VOCToolSet.extract_object(os.path.join(rt, VOCToolSet.id2xml(fid)))
            ocs_all = list()
            if options.IoUS:
                if len(ocs_old) != 0 and len(ocs) != 0:
                    thw_old = np.array([[oo['bbox']['ymin'], oo['bbox']['xmin'], oo['bbox']['ymax'] - oo['bbox']['ymin'], oo['bbox']['xmax'] - oo['bbox']['xmin']] for oo in ocs_old])
                    thw_now = np.array([[oo['bbox']['ymin'], oo['bbox']['xmin'], oo['bbox']['ymax'] - oo['bbox']['ymin'], oo['bbox']['xmax'] - oo['bbox']['xmin']] for oo in ocs])
                    ioum = calc_iou_matrix_thw(thw_old, thw_now)
                    drop_in_now = sorted(list(set(np.where(ioum > options.IoUST)[1].tolist())))
                    drop_in_now.reverse()
                    [ocs.pop(i) for i in drop_in_now]
                else:
                    ocs = ocs
                pass
            ocs_all = ocs_old + ocs
            VOCToolSet.append_object(new_xml_dir, ocs_all)
            pass
        else:
            print('{0} is not found in {1}'.format(fid, rt))
        pass
    pass

#python Putil/tools/data_process/split.py --voc_statistic_file /root/workspace/data/1229-fight-run-jump/run/voc_xml_statistic.csv --set_rate 0.8 0.2 --set_name train val --target_class run --image_root /root/workspace/data/1229-fight-run-jump/run/JPEGImages/