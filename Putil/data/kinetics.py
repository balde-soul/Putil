# coding=utf-8
import Putil.data.common_data as pcd

class KineticsData(pcd.CommonDataForTrainEvalTest):
    def __init__(self, kinetics_root, use_rate, action_ids, remain_strategy):
        pcd.CommonDataWithAug.__init__(self, use_rate=use_rate, sub_data=action_ids, remain_strategy=remain_strategy)
        self._kinetics_root = kinetics_root
        pass
    pass