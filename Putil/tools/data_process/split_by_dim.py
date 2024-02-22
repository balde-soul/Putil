##@FileName split.py
# @Note
# @Author cjh
# @Time 2022-12-10
#In[]
import cvxpy, argparse, os, json, sys
import random, sklearn
random.seed(1990)
sklearn.random.seed(1990)
import numpy as np
np.random.seed(1990)
import pandas as pd
from sklearn.utils import shuffle
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

##@brief 对数据集进行分割,保证目标属性均匀分割
# @note
# @time 2023-03-07
# @author cjh
class SplitData:
    ##@brief
    # @note
    # @param[in] property_dims 目标属性维度名称列表,pd.DataFrame的columns
    # @return 
    # @time 2023-03-07
    # @author cjh
    def __init__(self, property_dims):
        self._property_dims = property_dims
        pass

    ##@brief 
    # @note
    # @param[in]
    # @param[in] data, pd.DataFrame,每一行作为一个数据
    # @return 
    # @time 2023-03-07
    # @author cjh
    def split(self, data):
        extracted_property = data[self._property_dims]
        x = cvxpy.Variable(len(extracted_property), boolean=True)
        pass
    pass
data = pd.DataFrame([
    [2, 'data1', 1, 1, 2], 
    [3, 'data2', 0, 1, 2], 
    [1, 'data3', 2, 3, 1]
    ], 
    columns=['n', 'data_label', 'A', 'B', 'C']
    )
sd = SplitData(property_dims=['A', 'B', 'C'])
sd.split(data)

#parser = argparse.ArgumentParser()
#parser.add_argument('--data_list', dest='DataList', type=str, default=None, help='指定voc_statistic_file中的子集')
#parser.add_argument('--set_rate', dest='SetRate', type=float, nargs='+', help='指定分割比例')
#parser.add_argument('--set_name', dest='SetName', type=str, nargs='+', help='指定分割集合名称')
#parser.add_argument('--target_class', dest='TargetClass', type=str, nargs='+', help='指定目标名')
#parser.add_argument('--image_root', dest='ImageRoot', type=str, default='', help='图像根目录')
#parser.add_argument('--attachabled', dest='Attachabled', action='store_true', help='是否链接debug')
#options = parser.parse_args()
#if options.Attachabled:
#    import ptvsd
#    host = '127.0.0.1'
#    port = 12345
#    ptvsd.enable_attach(address=(host, port), redirect_output=True)
#    ptvsd.wait_for_attach()
#    pass
##options.VOCStatisticFile = '/root/workspace/data/chef-uniform/voc_xml_statistic.csv'
##options.SetRate = [0.8, 0.2]
##options.SetName = ['train', 'test']
##options.TargetClass = ['chef uniform', 'no chef uniform']
##options.ImageRoot = '/root/workspace/data/chef-uniform/images'
##options.DataList = '/root/workspace/data/chef-uniform/train.txt'
#
#if not os.path.exists(options.VOCStatisticFile):
#    print('voc statistic file does not exist')
#    sys.exit(1)
#    pass
#
#df = pd.read_csv(options.VOCStatisticFile)
#
## 使用data_list过滤目标，一般用于统一statistic file多次分割
#if options.DataList is not None:
#    with open(options.DataList, 'r') as fp:
#        sl = [os.path.split(i.replace('\n', ''))[-1] for i in fp.readlines()]
#        df = df[df['image_id'].apply(lambda x: x in sl)]
#        pass
#    pass
#
#def format_func(x):
#    types = json.loads(x['type_set'])
#    x['type_set'] = types
#    for _type in x['type_set']:
#        x[_type] = json.loads(x[_type])
#        pass
#    return x
#
#df = df.apply(format_func, axis=1)
#
#def calc_class_num(x):
#    ret = {tc: 0 for tc in options.TargetClass}
#    for tc in options.TargetClass:
#        if tc in x['type_set']:
#            ret[tc] = len(x[tc])
#            pass
#        pass
#    return pd.Series(ret)
#
#df = shuffle(df)
#class_num_df = df.apply(calc_class_num, axis=1)
#
#L = class_num_df.to_numpy().T
#
#nc, n = L.shape
#
#train_mins = (np.sum(L, axis=1, keepdims=False) * options.SetRate[0]).astype(np.int32)
#valid_mins = (np.sum(L, axis=1, keepdims=False) * options.SetRate[1]).astype(np.int32)
#
#x = cvxpy.Variable(n, boolean=True)
#lr = cvxpy.Variable(nc, nonneg=True)
#ur = cvxpy.Variable(nc, nonneg=True)
#
#lb = (L @ x >= train_mins.T - lr)
#ub = (L @ x <= (sum(L.T) - valid_mins).T + ur)
#constraints = [lb, ub]
#
#objective = (sum(lr) + sum(ur)) ** 2
#
#problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
#problem.solve()
#
#'''
#cvxpy.error.SolverError:
#
#                    You need a mixed-integer solver for this model. Refer to the documentation
#                        https://www.cvxpy.org/tutorial/advanced/index.html#mixed-integer-programs
#                    for discussion on this topic.
#
#                    Quick fix 1: if you install the python package CVXOPT (pip install cvxopt),
#                    then CVXPY can use the open-source mixed-integer linear programming
#                    solver `GLPK`. If your problem is nonlinear then you can install SCIP
#                    (pip install pyscipopt).
#
#                    Quick fix 2: you can explicitly specify solver='ECOS_BB'. This may result
#                    in incorrect solutions and is not recommended.
#'''
##In[]
#problem.status
#result = x.value
#
#index = np.array([int(round(i)) for i in result])
#
#train_index = np.where(index == 1)[0]
#val_index = np.where(index == 0)[0]
#
#with open(os.path.join(os.path.dirname(options.VOCStatisticFile), '{0}.txt'.format(options.SetName[0])), 'w') as fp:
#    for i in df.iloc[train_index]['image_id']:
#        fp.write('{0}\n'.format(os.path.join(options.ImageRoot, i)))
#        pass
#    pass
#
#with open(os.path.join(os.path.dirname(options.VOCStatisticFile), '{0}.txt'.format(options.SetName[1])), 'w') as fp:
#    for i in df.iloc[val_index]['image_id']:
#        fp.write('{0}\n'.format(os.path.join(options.ImageRoot, i)))
#        pass
#    pass
#
#train_sum = np.sum(class_num_df.to_numpy()[train_index], axis=0)
#val_sum = np.sum(class_num_df.to_numpy()[val_index], axis=0)
#print('{2} {3}_rate: {0}, {4}_rate: {1}'.format(train_sum/(train_sum + val_sum), val_sum/(train_sum + val_sum), class_num_df.columns, options.SetName[0], options.SetName[1]))