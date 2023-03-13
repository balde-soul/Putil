# coding=utf-8
import os, sys, gif
import numpy as np
#from mmaction2.mmaction.models.skeleton_gcn.skeletongcn import SkeletonGCN
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import Putil.data.human36m as Human36M
import Putil.data.aug as pAug
import Putil.visual.graph_view as GV
import matplotlib.pyplot as plt
import ptvsd
host = '127.0.0.1'
port = 65535
ptvsd.enable_attach(address=(host, port), redirect_output=True)
#ptvsd.wait_for_attach()

data = Human36M.Human36MData(
    data_root=r'/data/caojihua/data/Human3.6M/',
    stage=Human36M.Human36MData.Stage.Train,
    use_rate=1.0,
    subjects=Human36M.Human36MData.Subject.S1,
    scenarios=Human36M.Human36MData.Scenario.Walking,
    remain_strategy=Human36M.Human36MData.RemainStrategy.Drop,
    data_type=Human36M.Human36MData.DataType.Skeleton
)
root_node = pAug.AugNode(pAug.AugFuncNoOp())
Original = root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
root_node.freeze_node()
data.set_aug_node_root(root_node, [1.0])

a = data[0]
skeletondata = a[1][0].reshape([a[1][0].shape[0], -1, 2])
skeletondata[:, :, 1] = -skeletondata[:, :, 1]

arms = [21, 19, 18, 17, 13, 25, 26, 27, 29]
rightHand = [19, 22] # 11 <-> 24
leftHand = [27, 30] # 7 <-> 22
legs = [5, 4, 3, 2, 1, 0, 6, 7, 8, 9, 10]
body = [15, 14, 13, 12, 11, 0]  # 3 <-> 2 <-> 20 ...

chains = [arms, rightHand, leftHand, legs, body]

amax = max(skeletondata.max(axis=(0, 1)))
amin = min(skeletondata.min(axis=(0, 1)))

frames = []
for i, points in enumerate(skeletondata[0: 1000]):
    fig = plt.figure(8)
    ax = plt.gca()
    plt.axis([amin, amax, amin, amax])
    of = GV.gif_frame(fig, ax, chains, points)
    frames.append(of)
gif.save(frames, 'humanvisual.gif', duration=10)