# coding=utf-8
import os, sys, re
import Putil.data.common_data as pcd
import numpy as np
from enum import Flag

def _print_toolbar(rate, annotation=''):
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')

def _end_toolbar():
    sys.stdout.write('\n')

def _load_missing_file(path):
    missing_files = dict()
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1]
            if line not in missing_files:
                missing_files[line] = True 
    return missing_files 

def ntu_read_skeleton(file_path):
    f = open(file_path, 'r')
    datas = f.readlines()
    f.close()
    max_body = 4
    njoints = 25

    # specify the maximum number of the body shown in the sequence, according to the certain sequence, need to pune the 
    # abundant bodys. 
    # read all lines into the pool to speed up, less io operation. 
    nframe = int(datas[0][:-1])
    bodymat = dict()
    bodymat['file_name'] = file_path[-29:-9]
    nbody = int(datas[1][:-1])
    bodymat['nbodys'] = [] 
    bodymat['njoints'] = njoints 
    for body in range(max_body):
        bodymat['skel_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 3))
        bodymat['rgb_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
        bodymat['depth_body{}'.format(body)] = np.zeros(shape=(nframe, njoints, 2))
    # above prepare the data holder
    cursor = 0
    for frame in range(nframe):
        cursor += 1
        bodycount = int(datas[cursor][:-1])    
        if bodycount == 0:
            continue 
        # skip the empty frame 
        bodymat['nbodys'].append(bodycount)
        for body in range(bodycount):
            cursor += 1
            skel_body = 'skel_body{}'.format(body)
            rgb_body = 'rgb_body{}'.format(body)
            depth_body = 'depth_body{}'.format(body)
            
            bodyinfo = datas[cursor][:-1].split(' ')
            cursor += 1
            
            njoints = int(datas[cursor][:-1])
            for joint in range(njoints):
                cursor += 1
                jointinfo = datas[cursor][:-1].split(' ')
                jointinfo = np.array(list(map(float, jointinfo)))
                bodymat[skel_body][frame, joint] = jointinfo[:3]
                bodymat[depth_body][frame, joint] = jointinfo[3:5]
                bodymat[rgb_body][frame, joint] = jointinfo[5:7]
    # prune the abundant bodys 
    for each in range(max_body):
        try:
            if len(bodymat['nbodys']) == 0 or each >= max(bodymat['nbodys']):
                del bodymat['skel_body{}'.format(each)]
                del bodymat['rgb_body{}'.format(each)]
                del bodymat['depth_body{}'.format(each)]
        except Exception as ex:
            print('a')
    return bodymat 

class NTURGBDData(pcd.StandardCommonData):
    SkeletonsFold = 'nturgb+d_skeletons'
    class DataType(Flag):
        Skeleton=0x01
        FullDepthMaps=0x02
        RGBVideos=0x04
        IRData=0x08
        MaskedDepthMaps=0x0f
    
    class ActionType(Flag):
        pass
    ##@brief
    # @note
    # @param[in] ntu_root
    # @param[in]
    # @return 
    # @time 2023-02-28
    # @author cjh
    def __init__(
        self,
        ntu_root,
        use_rate,
        remain_strategy,
        camera_angles=None,
        camera_ids=None,
        person_ids=None,
        action_ids=None,
        data_type=DataType.Skeleton | DataType.FullDepthMaps | DataType.RGBVideos | DataType.IRData | DataType.MaskedDepthMaps
        ):
        sub_data = {
            'camera_angles': camera_angles,
            'camera_ids': camera_ids,
            'person_ids': person_ids,
            'action_ids': action_ids
        }
        self._ntu_root = ntu_root
        self._data_type = data_type
        pcd.StandardCommonData.__init__(self, use_rate=use_rate, sub_data=sub_data, remain_strategy=remain_strategy)
        pass

    def _id_to_action_id(self, _id):
        find = re.search('A', _id)
        return _id[find.end(): find.end() + 3]

    def _id_to_person_id(self, _id):
        find = re.search('P', _id)
        return _id[find.end(): find.end() + 3]
    
    def _id_to_camera_id(self, _id):
        find = re.search('C', _id)
        return _id[find.end(): find.end() + 3]

    def _sub_data_filteration(self, sub_data, remain_strategy):
        ret = list()
        for _id in self._data_field:
            if False not in [
                sub_data['action_ids'] is None or self._id_to_action_id(_id) in sub_data['action_ids'],
                sub_data['camera_ids'] is None or self._id_to_camera_id(_id) in sub_data['camera_ids'],
                sub_data['person_ids'] is None or self._id_to_person_id(_id) in sub_data['person_ids']
            ]:
                ret.append(_id)
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
        if self._data_type & NTURGBDData.DataType.Skeleton:
            _skeleton_field = set([i.replace('.skeleton', '') for i in os.listdir(os.path.join(self._ntu_root, NTURGBDData.SkeletonsFold))])
            finall_set = _skeleton_field
        else:
            raise NotImplementedError('当前数据类型未支持')
        return list(finall_set)

    ##@brief 通过index读取数据集合id返回目标数据
    # @note 不在目标数据中的类型为None
    # @param[in]
    # @param[in]
    # @return 目标集合数据格式为tuple，每个元素的意义为(action_id, skeleton,)
    # @time 2023-03-02
    # @author cjh
    def _generate_from_origin_index(self, index):
        _id = self._data_field[index]
        info = {
            'ID': _id,
            'action_id': self._id_to_action_id(_id),
            'camera_id': self._id_to_camera_id(_id),
            'person_id': self._id_to_person_id(_id),
        }
        return (
            info,
            ntu_read_skeleton(os.path.join('{0}{1}{2}'.format(self._ntu_root, os.path.sep, NTURGBDData.SkeletonsFold), '{0}.skeleton'.format(_id))),
        )

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