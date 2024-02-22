# coding=utf-8
import sys, os, argparse, random
from enum import Flag
import numpy as np
import pandas as pd
import pickle as pkl
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

parser = argparse.ArgumentParser()
parser.add_argument('--in_pkls', dest='InPkls', type=str, nargs='+', default=[], help='指定想要合并的pkl文件')
parser.add_argument('--out_pkl', dest='OutPkl', type=str, action='store', default='', help='指定合并输出的文件路径')
parser.add_argument('--pkl_type', dest='PklType', type=str, action='store', default='', help='指定pkl中保存的数据类型，仅支持几种类型:\{list\}')
parser.add_argument('--remain_pkl', dest='RemainPkl', type=str, action='store', default='', help='指定采样剩余的样本保存到文件，如果不指定，则丢弃剩余')
parser.add_argument('--do_balance', dest='DoBalance', action='store_true', help='指定时，进行样本平衡')
options = parser.parse_args()
#options.InPkls = ['/data/caojihua/data/0228run/KeypointV1-ForSport.pkl', 
#'/data/caojihua/data/0228jump/KeypointV1-ForSport.pkl', 
#'/data/caojihua/data/0301jump/KeypointV1-ForSport.pkl', 
#'/data/caojihua/data/0302run/KeypointV1-ForSport.pkl', 
#'/data/caojihua/data/NTU-RGBD/KeypointV1-ForSport.pkl', 
#'/data/caojihua/data/Human3.6M/KeypointV1-ForSport.pkl']
#options.OutPkl = '/data/caojihua/data/TsportSkeleton/Testtrain.pkl'
#options.PklType = 'list'
#options.RemainPkl = '/data/caojihua/data/TsportSkeleton/Testtest.pkl'

class PklType(Flag):
    list=1

if PklType[options.PklType] not in PklType:
    print('unsupported type')
    sys.exit(1)

if len(options.InPkls) == 0:
    print('no input pkl file')
    sys.exit(1)
else:
    for f in options.InPkls:
        if not os.path.exists(f):
            print('{0} does not exist'.format(f))
            sys.exit(1)
        pass
    pass

if not os.path.exists(os.path.dirname(options.OutPkl)):
    print('output dir: {0} does not exist'.format(os.path.dirname(options.OutPkl)))
    sys.exit(1)

def statistic_label(data):
    all_type = [d['label'] for d in data]
    type_set = set(all_type)
    all_type_array = np.array(all_type)
    type_dict = dict()
    for ts in type_set:
        a = np.where(all_type_array == ts)[0].tolist()
        random.shuffle(a)
        type_dict[ts] = a
    return type_dict

def print_status(data):
    data_field = dict()
    all_class = [i['label'] for i in data]
    for c in set(all_class):
        data_field[c] = (np.array(all_class) == c).sum()
        pass
    print('target: {0}'.format(data_field))
    pass

if PklType[options.PklType] is PklType.list:
    if options.DoBalance:
        statistic = []
        data = dict()
        for f in options.InPkls:
            with open(f, 'rb') as fp:
                data[f] = pkl.load(fp)
                type_dict = statistic_label(data[f])
                type_dict['file_name'] = f
                statistic.append(type_dict)
                pass
            pass
        sdf = pd.DataFrame(statistic)
        sdf = sdf.set_index('file_name')
        sdf = sdf.where((sdf.notna()), None)
        def type_sum(x):
            amount = 0
            for n in x:
                amount += len(n) if n is not None else 0
            return amount
        type_rate = sdf.apply(type_sum, axis=0).min() / sdf.apply(type_sum, axis=0)
        def do_sample(x):
            ret = list()
            remain = list()
            for _type, se in x.items():
                if se is None:
                    ret += []
                else:
                    #ret += np.random.choice(se, int(len(se) * type_rate[_type])).tolist()
                    ret += se[0: int(len(se) * type_rate[_type])]
                    remain += se[int(len(se) * type_rate[_type]): -1]
                    pass
                pass
            return pd.Series({'target': ret, 'remain': remain})
        target = sdf.apply(do_sample, axis=1)
        ret = list()
        for f, i in target['target'].items():
            ret += np.array(data[f])[i].tolist()
            pass
        print_status(ret)
        with open(options.OutPkl, 'wb') as fp:
            pkl.dump(ret, fp)
            pass
    
        if options.RemainPkl != '':
            ret = list()
            for f, i in target['remain'].items():
                ret += np.array(data[f])[i].tolist()
                pass
            print_status(ret)
            with open(options.RemainPkl, 'wb') as fp:
                pkl.dump(ret, fp)
                pass
        pass
    else:
        ret = list()
        for f in options.InPkls:
            with open(f, 'rb') as fp:
                ret += pkl.load(fp)
                pass
            pass
        print_status(ret)
        with open(options.OutPkl, 'wb') as fp:
            pkl.dump(ret, fp)
            pass
        pass