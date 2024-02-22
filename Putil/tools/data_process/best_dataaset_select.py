##@FileName split.py
# @Note
# @Author cjh
# @Time 2022-12-10
#In[]
import numpy as np
import pandas as pd
import cvxpy, argparse, os, json, sys

parser = argparse.ArgumentParser()
parser.add_argument('--statistic_file', dest='StatisticFile', type=str, action='store', default='', help='指定统计文件')
parser.add_argument('--target_class', dest='TargetClass', type=str, nargs='+', help='指定目标名')
options = parser.parse_args()
root_dir = os.path.dirname(options.VOCStatisticFile)

if not os.path.exists(options.StatisticFile):
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

def calc_class_num(x):
    ret = {tc: 0 for tc in options.TargetClass}
    for tc in options.TargetClass:
        if tc in x['type_set']:
            ret[tc] = len(x[tc])
            pass
        pass
    return pd.Series(ret)

class_num_df = df.apply(calc_class_num, axis=1)

L = class_num_df.to_numpy().T

#image_num = 1000
#class_num = 10
#
#L = np.reshape(np.random.random(image_num * class_num) * 5, [class_num, image_num]).astype(np.int32)
#print(L)

#L = np.array([[2,3,0,3,0,8,4],
#              [3,0,2,0,0,0,4],
#              [0,2,5,1,3,0,2]])
nc, n = L.shape

#train_mins = np.array([12, 6, 8])
#valid_mins = np.array([7, 3, 4])
train_mins = (np.sum(L, axis=1, keepdims=False) * 0.8).astype(np.int32)
valid_mins = (np.sum(L, axis=1, keepdims=False) * 0.2).astype(np.int32)

x = cvxpy.Variable(n, boolean=True)
lr = cvxpy.Variable(nc, nonneg=True)
ur = cvxpy.Variable(nc, nonneg=True)

lb = (L @ x >= train_mins.T - lr)
ub = (L @ x <= (sum(L.T) - valid_mins).T + ur)
constraints = [lb, ub]

objective = (sum(lr) + sum(ur)) ** 2

problem = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
problem.solve()

'''
cvxpy.error.SolverError:

                    You need a mixed-integer solver for this model. Refer to the documentation
                        https://www.cvxpy.org/tutorial/advanced/index.html#mixed-integer-programs
                    for discussion on this topic.

                    Quick fix 1: if you install the python package CVXOPT (pip install cvxopt),
                    then CVXPY can use the open-source mixed-integer linear programming
                    solver `GLPK`. If your problem is nonlinear then you can install SCIP
                    (pip install pyscipopt).

                    Quick fix 2: you can explicitly specify solver='ECOS_BB'. This may result
                    in incorrect solutions and is not recommended.
'''
#In[]
problem.status
result = x.value

index = np.array([int(round(i)) for i in result])

train_index = np.where(index == 1)[0]
val_index = np.where(index == 0)[0]

with open(os.path.join(os.path.dirname(options.VOCStatisticFile), 'train.txt'), 'w') as fp:
    for i in df.iloc[train_index]['image_id']:
        fp.write('{0}\n'.format(os.path.join(options.ImageRoot, i)))
        pass
    pass

with open(os.path.join(os.path.dirname(options.VOCStatisticFile), 'val.txt'), 'w') as fp:
    for i in df.iloc[val_index]['image_id']:
        fp.write('{0}\n'.format(os.path.join(options.ImageRoot, i)))
        pass
    pass

train_sum = np.sum(class_num_df.to_numpy()[train_index], axis=0)
val_sum = np.sum(class_num_df.to_numpy()[val_index], axis=0)
print('{2} train_rate: {0}, val_rate: {1}'.format(train_sum/(train_sum + val_sum), val_sum/(train_sum + val_sum), class_num_df.columns))