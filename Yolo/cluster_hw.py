# coding = utf-8

import json
from sklearn import cluster as clu
import pandas as pd
import numpy as np
from colorama import Fore
import Putil.calc.estimate as es
import matplotlib.pyplot as plt
import os

cluseter_type = ['k-means']


class yolo_cluster:
    """
    yolo矩形框先验高宽计算
    多种类型聚类-->生成item-mIoU曲线
    生成pd文件：
             cluseter_count_1     cluseter_count_2

        type      H, W              H, W||H, W
    """
    def __init__(self, array, **options):
        """
        options: max_cluster: limit the max cluster num for the cluster
        options: result_path: the path where saving the result
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

    def __kmeans(self, iou_image_name, distribution_image_name):
        miou = list()
        # fig = plt.figure()
        # plt.subplots(121)
        acc_cluster = list()
        for i in range(1, self._MaxCluster + 1):
            try:
                kmean = clu.KMeans(n_clusters=i, precompute_distances='auto', algorithm='auto', random_state=0)
                kmean.fit(self._GT)
                store_h_w_str = ''
                for j in kmean.cluster_centers_:
                    store_h_w_str += '||' + str(j[0]) + ',' + str(j[1])
                self.TotalPd.loc['k-means', i] = store_h_w_str[2: ]
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
                acc_cluster.append(i)
                self._IoUPd.loc['k-means', i] = np.mean(m_iou)
                pass
            except Exception:
                print(Fore.RED + '{0} cluster is not run({1})'.format(i, Exception.args))
                pass
        plt.plot(acc_cluster, miou)
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

    def __cluster(self, c_type,  iou_image_name, distribution_image_name, **options):
        if c_type == 'k-means':
            self.__kmeans(iou_image_name, distribution_image_name)
            pass
        pass

    def analysis(self, *c_type, **options):
        iou_image_name = options.pop('iou_image_name', 'n_cluster-mIoU.png')
        distribution_image_name = options.pop('distribution_image_name', 'gt-hw-distribution.png')
        h_w_iou_file_name = options.pop('h_w_iou_file_name', 'h-w-analysis.xlsx')
        for i in c_type:
            assert i in cluseter_type, Fore.RED + i + 'is not supported'
        for _c_type in c_type:
            self.__cluster(_c_type, iou_image_name, distribution_image_name)
            pass
        writer = pd.ExcelWriter(os.path.join(self._AnalysisResultPath, h_w_iou_file_name))
        self.TotalPd.to_excel(writer, sheet_name='h-w')
        self._IoUPd.to_excel(writer, sheet_name='iou')
        writer.save()
        pass


# extract the prior height and width from the h_w_iou_file
def extract_prior(h_w_iou_file, **options):
    num = options.pop('num', None)
    assert (num is None) or (type(num) != type(int)), Fore.RED + 'num not support'
    _type = options.pop('cluster_type', 'k-means')
    prior_h = list()
    prior_w = list()
    # specify num , get prior_h, prior_w
    if num is not None:
        data_sheet = pd.read_excel(h_w_iou_file, sheet_name='h-w', index_col=0)
        prior_h_w = data_sheet[num][_type]
        count = 0
        split = prior_h_w.split("||")
        for one in split:
            s = one.split(',')
            prior_h.append(float(s[0]))
            prior_w.append(float(s[1]))
            pass
        return prior_h, prior_w
    # todo: if num has not been specified,
    # todo: we use the iou shhet to get the max iou and the get the prior_h and prior_w
    else:
        print(Fore.RED + 'this method is not conform')
        h_w_sheet = pd.read_excel(h_w_iou_file, sheet_name='h-w', index_col=0)
        iou_sheet = pd.read_excel(h_w_iou_file, sheet_name='h-w', index_col=0)
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
