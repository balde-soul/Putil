# coding=utf-8
import sys, os, argparse, yaml, re, json
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--data_cnf', dest='DataCnf', type=str, action='store', default='', help='指定yolo的data.yaml路径')
parser.add_argument('--statistic_save_to', dest='StatisticSaveTo', type=str, action='store', default='', help='统计过程中产生的中间文件保存路径')
options = parser.parse_args()

if not os.path.exists(options.DataCnf) or options.DataCnf == '':
    print('data_cnf: {0} not exist'.format(options.DataCnf))
    sys.exit(1)
    pass

try:
    with open(options.DataCnf, 'r') as fp:
        cnf = yaml.load(fp, Loader=yaml.FullLoader)
        train_list = cnf.get('train', '')
        val_list = cnf.get('val', '')
        test_list = cnf.get('test', '')
        classes = {i: c for i, c in enumerate(cnf.get('names', []))}
except Exception as ex:
    print(ex.args)
    raise ex
    pass

def statistic(train_list):
    dfd = list()
    with open(train_list, 'r') as fp:
        files = [f.replace('\n', '') for f in fp.readlines()]
        pass
    for tl in files:
        dfdd = dict()
        fid = os.path.basename(tl)
        s = re.search('\.', fid[::-1])
        fid = fid[::-1][s.end(): ][::-1]
        dfdd['fid'] = fid
        txt_dir = os.path.join(os.path.dirname(os.path.dirname(tl)), 'labels/{0}.txt'.format(fid))
        if os.path.exists(txt_dir):
            dfdd['txt_exist'] = 1
            try:
                df = pd.read_csv(txt_dir, sep=' ', header=None)
                dfdd['has_content'] = 1
                extend = list()
                for k in set(df[0]):
                    if k in classes.keys():
                        dfdd[classes[k]] = (df[0] == k).sum()
                    else:
                        extend.append(k)
                    pass
                dfdd['extend'] = json.dumps(extend)
            except pd.errors.EmptyDataError as ex:
                dfdd['has_content'] = 0
                pass
            pass
        else:
            dfdd['txt_exist'] = 0
            pass
        dfd.append(dfdd)
        pass
    df = pd.DataFrame(dfd)
    return df
    pass

if train_list != '':
    df = statistic(train_list)
    print('train: {0}'.format({v: df[v].sum() for k, v in classes.items()}))
    pass
if val_list != '':
    df = statistic(val_list)
    print('val: {0}'.format({v: df[v].sum() for k, v in classes.items()}))
    pass
if test_list != '':
    df = statistic(test_list)
    print('test: {0}'.format({v: df[v].sum() for k, v in classes.items()}))
    pass
else:
    print('train list is not specified in {0}'.format(options.DataCnf))