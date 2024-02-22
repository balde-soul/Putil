# coding=utf-8
import os, sys, re, cdflib
import Putil.data.common_data as pcd
import numpy as np
from enum import Flag
import h5py

##@brief
# @note 
# Human3.6M数据集的概况
# data: 代表着data_field中的元素
# id: 代表着文件名称，由动作名称与唯一录制时间构成，区分了同一动作不同时间录制
# @time 2023-03-04
# @author cjh
class Human36MData(pcd.StandardCommonData):
    VideoFold='videos'
    SkeletonSubDir='MyPoseFeatures{0}D2_Positions/'.format(os.path.sep) # 相对于subject目录下2DSkeleton存储路径
    SkeletonPostfix='cdf' # 2DSkeleton文件的后缀
    SubjectConcatIdLabel = '-' # 作为subject与id组成data_field元素的分隔符
    class DataType(Flag):
        All=0xff
        Skeleton=0x01
        Skeleton3D=0x02
        Video=0x04
        pass
    
    ##@brief
    # @note # importance: 数据文件中存在同个动作两种命名的情况,因此从Scenario到id是不可能的
    # @time 2023-03-04
    # @author cjh
    class Scenario(Flag):
        All=0xffff
        Directions=0x0001
        Discussion=0x0002
        Eating=0x0004
        Greeting=0x0008
        Phoning=0x0010
        Posing=0x0020
        Buying=0x0040
        Purchases=0x0040
        Sitting=0x0080
        SittingDown=0x0100
        Smoking=0x0200
        TakingPhoto=0x0400
        Photo=0x0400
        Waiting=0x0800
        Walking=0x1000
        WalkDog=0x2000
        WalkingDog=0x2000
        WalkTogether=0x4000
        pass

    class Subject(Flag):
        All=0xffff
        S1=0x0001
        S2=0x0002
        S3=0x0004
        S4=0x0008
        S5=0x0010
        S6=0x0020
        S7=0x0040
        S8=0x0080
        S9=0x0100
        S10=0x0200
        S11=0x0400
        pass

    def _data_to_scenario(self, data):
        _, _id = self._data_to_subject_and_id(data)
        return self._id_to_scenario(_id)

    def _id_to_scenario(self, _id):
        find_space = re.search(' ', _id)
        if  find_space is not None:
            scenario = _id[0: find_space.start()]
            return Human36MData.Scenario[scenario]
        else:
            find_point = re.search('\.', _id)
            assert find_point is not None
            scenario = _id[0: find_point.start()]
            return Human36MData.Scenario[scenario]

    ##@brief 从2DSkeleton文件名称中提取id
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-03-04
    # @author cjh
    def _skeleton_to_id(self, skl):
        return skl.replace('.{0}'.format(Human36MData.SkeletonPostfix), '')

    ##@brief 从id中获取2DSkeleton文件名称
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-03-04
    # @author cjh
    def _id_to_skeleton(self, _id):
        return '{0}.{1}'.format(_id, Human36MData.SkeletonPostfix)
    
    ##@brief data_field中的元素生成2DSkeleton文件路径
    # @note
    # @param[in]
    # @return 
    # @time 2023-03-04
    # @author cjh
    def _data_to_skeleton_file(self, data):
        subject, _id = self._data_to_subject_and_id(data)
        return os.path.join(self._data_root, os.path.join(
            '{0}{1}{2}'.format(subject.name, os.path.sep, Human36MData.SkeletonSubDir), 
            self._id_to_skeleton(_id)))
    
    ##@brief 传入subject获取2DSkeleton文件的保存路径
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-03-04
    # @author cjh
    def _subject_to_skeleton_dir(self, subject):
        return os.path.join(os.path.join(self._data_root, subject), Human36MData.SkeletonSubDir)

    ##@brief 使用subject与文件的id组成data_field中的元素
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-03-04
    # @author cjh
    def _subject_id_to_data(self, subject, _id):
        return '{0}{1}{2}'.format(subject, Human36MData.SubjectConcatIdLabel, _id)

    ##@brief 从data_field中的元素获取到subject和id
    # @note
    # @param[in] subjects: Human36MData.Subject
    # @param[in] scenarios: Human36MData.Scenarios
    # @return 
    # @time 2023-03-04
    # @author cjh
    def _data_to_subject_and_id(self, data):
        subject, _id = data.split(Human36MData.SubjectConcatIdLabel)
        return Human36MData.Subject[subject], _id

    def __init__(
        self,
        data_root,
        stage,
        use_rate,
        remain_strategy,
        subjects=Subject.All,
        scenarios=Scenario.All,
        data_type=DataType.All
        ):
        sub_data = {
            'subjects': subjects,
            'scenarios': scenarios,
        }
        self._data_root = data_root
        self._video_train_root = os.path.join(self._data_root, '{0}train256')
        self._data_type = data_type
        pcd.StandardCommonData.__init__(self, use_rate=use_rate, sub_data=sub_data, remain_strategy=remain_strategy)
        pass

    ##@brief todo: 通过sub_data过滤出目标数据集合
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-03-03
    # @author cjh
    def _sub_data_filteration(self, sub_data, remain_strategy):
        ret = list()
        for data in self._data_field:
            subject, _id = self._data_to_subject_and_id(data)
            scenrio = self._id_to_scenario(_id)
            if subject & sub_data['subjects'] and scenrio & sub_data['scenarios']:
                ret.append(data)
                pass
            pass
        return ret

    ##@brief 针对不通数据类型要求求取交集
    # @note
    # @param[in]
    # @param[in]
    # @return 
    # @time 2023-03-02
    # @author cjh
    def _generate_data_field(self):
        # 由于NTU有多种类型数据，同时针对一个id并不是所有类型数据都存在（比如skeleton数据就存在缺失），针对不通的目标数据类型，取其集合的交集
        if self._data_type & Human36MData.DataType.Skeleton:
            _data_field = list()
            for subject in os.listdir(self._data_root):
                if os.path.isdir(os.path.join(self._data_root, subject)):
                    for sk in os.listdir(self._subject_to_skeleton_dir(subject)):
                        _data_field.append(self._subject_id_to_data(subject, self._skeleton_to_id(sk)))
                        pass
                    pass
                pass
            finall_set = set(_data_field)
            pass
        else:
            raise NotImplementedError('当前数据类型未支持')
        return list(finall_set)

    ##@brief 通过index读取数据集合id返回目标数据
    # @note 不在目标数据中的类型为None
    # @param[in]
    # @param[in]
    # @return 目标集合数据格式为tuple，每个元素的意义为('skeleton',)
    # @time 2023-03-02
    # @author cjh
    def _generate_from_origin_index(self, index):
        data = self._data_field[index]
        meta = {
            'data': data,
            'scenario': self._data_to_scenario(data),
            'subject': self._data_to_subject_and_id(data)[0]
        }
        skeleton = self._read_skeleton(data)
        return (meta, skeleton,)

    def _read_skeleton(self, data):
        f = self._data_to_skeleton_file(data)
        cdf_file = cdflib.CDF(f)
        d = cdf_file.varget("Pose")
        return d

    def _restart_process(self, restart_param):
        '''
        process while restart the data, process in the derived class and called by restart_data
        restart_param: the argv which the derived class need, dict
        '''
        pass

    def _inject_operation(self, inject_param):
        '''
        operation while the epoch_done is False, process in the derived class and called by inject_operation
        injecct_param: the argv which the derived class need, dict
        '''
        pass
    pass