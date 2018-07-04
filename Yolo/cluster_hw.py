# coding = utf-8

import json
from sklearn import cluster as clu
import pandas as pd
import numpy as np
from colorama import Fore
import calc.estimate as es
import matplotlib.pyplot as plt
import os

cluseter_type = ['k-means']

class yolo_cluster:
    """
    yolo矩形框先验高宽计算
    多种类型聚类-->生成item-mIoU曲线
    生成pd文件：
             cluseter_count_1     cluseter_count_2

        type      H, W              H, W, H, W
    """
    def __init__(self, array, **options):
        """
        array:shape:[sampel_count, 2],[N, :] = [Height, Width]
        """
        self._GT = array
        self._MaxCluster = options.pop('max_cluster', 16)
        self._AnalysisResultPath = options.pop('result_path', './result')
        assert self._GT.shape[1] == 2, Fore.RED + 'shape should be [:, 2]'
        self.TotalPd = pd.DataFrame()
        self.TotalPd.insert(0, 'type', cluseter_type)
        for i in range(1, self._MaxCluster + 1):
            self.TotalPd.insert(len(self.TotalPd.loc[0]), i, value=0)
            pass
        self.TotalPd = self.TotalPd.set_index('type')
        self._IoUPd = pd.DataFrame()
        self._IoUPd.insert(0, 'type', cluseter_type)
        for i in range(1, self._MaxCluster + 1):
            self._IoUPd.insert(len(self._IoUPd.loc[0]), i, value=0)
            pass
        self._IoUPd = self._IoUPd.set_index('type')
        pass

    def __kmeans(self, **options):
        miou = list()
        # fig = plt.figure()
        # plt.subplots(121)
        for i in range(1, self._MaxCluster + 1):
            kmean = clu.KMeans(n_clusters=i, precompute_distances='auto', algorithm='auto', random_state=0)
            kmean.fit(self._GT)
            self.TotalPd.loc['k-means', i] = str(kmean.cluster_centers_).replace('[', '').replace(']', '').replace(',', ' ')
            m_iou = list()
            for j in self._GT:
                one_gt_iou = list()
                for k in kmean.cluster_centers_:
                    m_rect = np.concatenate([np.array([0, 0]), k], axis=0)
                    gt_rect = np.concatenate([np.array([0, 0]), j], axis=0)
                    one_gt_iou.append(es.calc_iou(m_rect, gt_rect, LHW=True))
                    pass
                m_iou.append(np.mean(one_gt_iou))
                pass
            # sca = plt.scatter(np.transpose(self._GT, [1, 0])[0], np.transpose(self._GT, [1, 0])[1], marker='.', )
            # plt.setp(sca, markersize=2)
            miou.append(np.mean(m_iou))
            self._IoUPd.loc['k-means', i] = np.mean(m_iou)
            pass
        plt.plot(list(range(1, self._MaxCluster + 1)), miou)
        plt.xlabel('n_cluster')
        plt.ylabel('mIoU')
        plt.savefig(os.path.join(self._AnalysisResultPath, 'n_cluster-mIoU.png'))
        plt.close()
        plt.scatter(np.transpose(self._GT, [1, 0])[0], np.transpose(self._GT, [1, 0])[1])
        plt.xlabel('H')
        plt.ylabel('W')
        plt.savefig(os.path.join(self._AnalysisResultPath, 'gt-hw-distribution.png'))
        plt.close()
        pass

    def __cluster(self, c_type,  **options):
        if c_type == 'k-means':
            self.__kmeans()
            pass
        pass

    def analysis(self, *c_type, **options):
        for i in c_type:
            assert i in cluseter_type, Fore.RED + i + 'is not supported'
        for _c_type in c_type:
            self.__cluster(_c_type)
            pass
        writer = pd.ExcelWriter(os.path.join(self._AnalysisResultPath, 'h-w-analysis.xlsx'))
        self.TotalPd.to_excel(writer, sheet_name='h-w')
        self._IoUPd.to_excel(writer, sheet_name='iou')
        writer.save()
    pass


def __test_yolo_cluster_init():
    data1 = np.random.random_sample(500) + 5
    data2 = np.random.random_sample(500)
    data = np.concatenate([data1, data2], axis=0)
    yolo_pre = yolo_cluster(data.reshape([500, 2]), result_path='../test/yolo/')
    yolo_pre.analysis('k-means')
    pass


if __name__ == '__main__':
    __test_yolo_cluster_init()
