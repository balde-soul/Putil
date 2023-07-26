# coding=utf-8
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import sys, os, cv2
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    import argparse, re
    import pandas as pd
    from Putil.tools.data_process.VOCStatistic import StatisticFileToolset
    from Putil.tools.data_process.voc_util import VOCToolSet
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_root', dest='ImageRoot', type=str, default='', action='store', help='指定图像根目录')
    parser.add_argument('--xml_root', dest='XmlRoot', type=str, default='', action='store', help='指定xml根目录')
    parser.add_argument('--extract_to', dest='ExtractTo', type=str, default='', help='指定提取目标位置')
    options = parser.parse_args()

    if options.XmlRoot == '' or options.ImageRoot == '' or options.ExtractTo == '':
        print('please specify the path')
        sys.exit(1)
    
    if not os.path.exists(options.ExtractTo) or os.path.isdir(options.ExtractTo):
        os.mkdir(options.ExtractTo)
    
    
    got_name = list()
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    for imgf in os.listdir(options.ImageRoot):
        imgdir = os.path.join(options.ImageRoot, imgf)
        imgid = VOCToolSet.image2id(imgdir)
        xmldir = os.path.join(options.XmlRoot, VOCToolSet.id2xml(imgid))
        if not os.path.exists(xmldir):
            continue
        objs = VOCToolSet.extract_object(xmldir)
        if len(objs) == 0:
            continue
        img = cv2.imread(imgdir)
        if img is None:
            continue
        for obji, obj in enumerate(objs):
            obj_name = obj['name']
            rectangle = [obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']]
            obj_to_dir = os.path.join(options.ExtractTo, obj_name)
            if obj_name not in got_name:
                os.mkdir(obj_to_dir)
                got_name.append(obj_name)
            cv2.imwrite(os.path.join(obj_to_dir, '{0}-{1}.jpg').format(imgid, obji), img[rectangle[1]: rectangle[3], rectangle[0]: rectangle[2], :], encode_param)