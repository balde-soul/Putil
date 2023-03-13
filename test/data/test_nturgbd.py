# coding=utf-8
import enum, os, sys, gif
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import Putil.data.nturgbd as NTURGBD
import Putil.data.aug as pAug
import Putil.visual.graph_view as GV
import ptvsd
host = '127.0.0.1'
port = 65535
#ptvsd.enable_attach(address=(host, port), redirect_output=True)
#ptvsd.wait_for_attach()

data = NTURGBD.NTURGBDData(
    ntu_root=r'/data/caojihua/data/NTU-RGBD/',
    use_rate=0.1,
    #action_ids=['099', '026', '027'],
    action_ids=['099'],
    camera_ids=['001'],
    remain_strategy=NTURGBD.NTURGBDData.RemainStrategy.Drop,
    data_type=NTURGBD.NTURGBDData.DataType.Skeleton
)
root_node = pAug.AugNode(pAug.AugFuncNoOp())
Original = root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
root_node.freeze_node()
data.set_aug_node_root(root_node, [1.0])

a = data[0]
skeletondata = a[1]['skel_body0'][:, :, 0: 2]
# 22: 右拇指 21右小指 24: 左拇指 23 左小指
#skeletondata = skeletondata[:, [3, 4, 8, 12, 16, 13, 17, 14, 18, 15, 19, 5, 6, 22, 21, 24, 23], :]

arms = [23, 11, 10, 9, 8, 20, 4, 5, 6, 7, 21] # 23 <-> 11 <-> 10 ...
#arms = [23, 10, 9, 8, 20, 4, 5, 6, 21] # 23 <-> 11 <-> 10 ...
rightHand = [11, 24] # 11 <-> 24
#rightHand = [10, 24] # 11 <-> 24
leftHand = [7, 22] # 7 <-> 22
#leftHand = [6, 22] # 7 <-> 22
legs = [19, 18, 17, 16, 0, 12, 13, 14, 15] # 19 <-> 18 <-> 17 ...
body = [3, 2, 20, 1, 0]  # 3 <-> 2 <-> 20 ...
chains = [arms, rightHand, leftHand, legs, body]
#chains = []

print(skeletondata.shape)
amax = max(skeletondata.max(axis=(0, 1)))
amin = min(skeletondata.min(axis=(0, 1)))

frames = []
for i, points in enumerate(skeletondata):
    fig = plt.figure(8)
    ax = plt.gca()
    plt.axis([amin, amax, amin, amax])
    plt.title(a[0])
    of = GV.gif_frame(fig, ax, chains, points, (0.01, 0.01))
    frames.append(of)
    pass
gif.save(frames, 'ntgvisual.gif', duration=10)