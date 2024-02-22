# coding=utf-8
#In[]
import json, os, sys, sympy, scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Putil.tools.data_process.voc_util import VOCToolSet
from Putil.calc.iou import calc_iou_matrix_thw

collect_file = '/data/caojihua/Project/yolov5/yolov5v6/runs/detect/TchefhatMV6351/exp/det_col.csv'
class_set = ["black_transparent_head_cover", "dh", "black_chefs_cap", "black_beret", "white_beret", "red_beret", 
"white_chefs_cap_high", "black_chefs_cap_high", "blue_beret", "cap_head", "no_cap_head", 
"blur_black_hat", "blur_no_cap_head"]
target_class = ["black_transparent_head_cover", "black_beret", "no_cap_head"]
bind_iou = 0.2
def get_xml_path(img_path):
    return os.path.join(os.path.dirname(os.readlink(os.path.dirname(os.readlink(img_path)))), 'Annotations-ForChefHat-Manualed')

def get_id(img_path):
    return VOCToolSet.image2id(os.readlink(img_path))

def get_label_bbox(img_path):
    xmldir = os.path.join(get_xml_path(img_path), VOCToolSet.id2xml(get_id(img_path)))
    objects = VOCToolSet.extract_object(xmldir)
    tcls = list()
    tboxes = list()
    for obj in objects:
        tcls.append(obj['name'])
        tboxes.append([obj['bbox']['xmin'], obj['bbox']['ymin'], obj['bbox']['xmax'], obj['bbox']['ymax']])
    return tboxes, tcls

df = pd.read_csv(collect_file, encoding='utf-8', index_col=None)

def unformated(x):
    for k, v in x.items():
        try:
            x[k] = json.loads(v)
        except Exception as ex:
            pass
    return x

df = df.apply(unformated, axis=1)

#In[]
feature_dfs = list()

for i in range(0, len(df)):
    tboxes, tcls = get_label_bbox(df.loc[i]['path'])
    if len(tboxes) != 0:
        ioum = calc_iou_matrix_thw(df.loc[i]['boxes'], tboxes)
        okbind = np.sum(ioum > bind_iou, axis=1) >= 1
        okindx = np.argmax(ioum, axis=1)[okbind]
        bindc_f = np.array(df.loc[i]['clsconf'])[okbind]
        bindc = np.array(tcls)[okindx]
        feature_dfs.append(pd.DataFrame(np.concatenate([np.expand_dims(bindc, axis=-1), bindc_f], axis=-1), columns=['class'] + list(range(0, len(class_set)))))
    pass
feature_df = pd.concat(feature_dfs)

for i in set(feature_df['class']):
    feature = np.array(feature_df[feature_df['class'].eq(i)][list(range(len(class_set)))]).astype(np.float32)
    M = np.sum(feature, axis=0, keepdims=True) / feature.shape[0]
    S = np.matmul(np.transpose((feature - M)), feature - M) / feature.shape[0]
    fig = plt.figure(figsize=(13, 12))
    ax = fig.add_subplot(2, 1, 1)
    ax.imshow(M, cmap='plasma')
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(S, cmap='plasma')
    plt.savefig('{0}.png'.format(i))

    # todo: get parameter
    # solver
    # todo: get fix
#In[]
ti = np.arange(0.0, 1.0, 0.005)
want_target = "no_cap_head"
for i in class_set:
    target = [i, want_target]
    target_index = [np.where(np.array(class_set) == t)[0][0] for t in target]
    bthc = feature_df[feature_df['class'].eq(want_target)][target_index]
    import matplotlib.pyplot as plot
    bthc = np.array(bthc, dtype=np.float32)
    plt.axis([0., 1., 0., 1.])
    plot.scatter(bthc[:, 0], bthc[:, 1], s=5)
    plot.title('{0}-sample({1})'.format('-'.join(target), bthc.shape[0]))
    plot.show()
# In[]
#from scipy.stats import bernoulli
# 
## 生成样本
#p_1 = 1.0 / 2  # 假设样本服从p为1/2的bernouli分布
#fp = bernoulli(p_1)  # 产生伯努利随机变量
#xs = fp.rvs(1000)   # 产生100个样本
#print(xs[:30])      # 看看前面30个
## [0 1 1 1 1 0 0 0 0 1 0 1 0 0 0 0 1 1 1 1 0 1 1 0 0 1 0 0 0 1]
#import sympy
#import numpy as np
# 
## 估计似然函数
#x, p, z = sympy.symbols('x p z', positive=True)
#phi = p**x*(1-p)**(1-x)   #分布函数
#L = np.prod([phi.subs(x, i) for i in xs])   # 似然函数
#print(L)
## p**52*(-p + 1)**48
#logL = sympy.expand_log(sympy.log(L))
#sol = sympy.solve(sympy.diff(logL, p), p)
#print(sol)