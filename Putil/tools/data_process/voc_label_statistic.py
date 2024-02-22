##@FileName voc_label_statistic.py
# @Note 统计voc标记相关项
# @Author cjh
# @Time 2023-04-03
# coding=utf-8
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    import argparse, re
    import pandas as pd
    from Putil.tools.data_process.VOCStatistic import StatisticFileToolset
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_statistic_files', dest='VOCStatisticFiles', type=str, nargs='+', default=[], help='指定VOC数据集统计存储文件集合')
    parser.add_argument('--save_to', dest='SaveTo', type=str, default='', help='指定统计结果保存路径')
    options = parser.parse_args()

    # <block_begin: for test
    # @time 2023-04-03
    # @author cjh
    root_dir = '/data/caojihua/data/TchefhatT/'
    options.VOCStatisticFiles = [os.path.join(root_dir, i) for i in os.listdir(root_dir) if re.search('voc_statistic.csv', i) is not None]
    options.SaveTo = root_dir
    # block_end: >

    assert len(options.VOCStatisticFiles) != 0, print('voc statistic files is empty')
    for i, vsf in enumerate(options.VOCStatisticFiles):
        pass

    sfts = StatisticFileToolset()
    info = defaultdict(lambda: defaultdict(list))
    size_set = defaultdict(list)
    height_set = defaultdict(list)
    width_set = defaultdict(list)
    xset = defaultdict(list)
    yset = defaultdict(list)
    for i, vsf in enumerate(options.VOCStatisticFiles):
        df = sfts.read_statistic_file(vsf)
        if df.empty:
            continue
        ts = sfts.get_type_set(df)

        def get_size(x):
            for t in ts:
                if isinstance(x[t], list):
                    [info[t]['size_set'].append((bbox[3] - bbox[1]) * (bbox[2] - bbox[0])) for bbox in x[t]]
                    [info[t]['height_set'].append(bbox[3] - bbox[1]) for bbox in x[t]]
                    [info[t]['width_set'].append(bbox[2] - bbox[0]) for bbox in x[t]]
                    [info[t]['xset'].append(bbox[2] + bbox[0] / 2) for bbox in x[t]]
                    [info[t]['yset'].append(bbox[3] + bbox[1] / 2) for bbox in x[t]]
                    pass
                pass
            pass

        df.apply(get_size, axis=1)
    for ssk, ssv in info.items():
        plt.figure()
        hc, wc, scatter_size = 2, 2, 10
        plt.subplot(hc, wc, 1)
        plt.hist(ssv['size_set'], bins=100)
        plt.title('size')
        plt.subplot(hc, wc, 2)
        plt.scatter(ssv['height_set'], ssv['width_set'], s=scatter_size)
        plt.xlabel('height')
        plt.ylabel('height')
        plt.title('height-width')
        plt.subplot(hc, wc, 3)
        plt.scatter(ssv['xset'], ssv['yset'], s=scatter_size)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('center')
        plt.savefig(os.path.join(options.SaveTo, '{0}.jpg'.format(ssk)))