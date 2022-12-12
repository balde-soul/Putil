# coding=utf-8
import pandas as pd
import os, copy, json, random, sys
import argparse
random.seed(1995)

parser = argparse.ArgumentParser()
parser.add_argument('--voc_statistic_file', dest='VOCStatisticFile', type=str, default='', help='指定VOC数据集统计存储文件')
options = parser.parse_args()
options.VOCStatisticFile = '/root/workspace/data/chefhat/VOC2012/voc_xml_statistic.csv'

if not os.path.exists(options.VOCStatisticFile):
    print('voc statistic file does not exist')
    sys.exit(1)
    pass

df = pd.read_csv(options.VOCStatisticFile)

def format_func(x):
    types = json.loads(x['type_set'])
    x['type_set'] = types
    for _type in x['type_set']:
        x[_type] = json.loads(x[_type])
        pass
    return x

df = df.apply(format_func, axis=1)
print(df)