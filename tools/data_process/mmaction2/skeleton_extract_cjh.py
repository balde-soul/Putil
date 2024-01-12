# Copyright (c) OpenMMLab. All rights reserved.
import abc, argparse, enum, os, shutil, string, cv2, mmcv, sys, gif
import numpy as np
import matplotlib.pyplot as plt
from math import floor
from collections import defaultdict
import os.path as osp
import random as rd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import Putil.visual.graph_view as GV
from Putil.tools.data_process.voc_util import VOCToolSet
import Putil.data.nturgbd as NTURGBD
import Putil.data.aug as pAug
import Putil.visual.graph_view as GV
try:
    from mmdet.apis import inference_detector, init_detector
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_detector` and '
                      '`init_detector` form `mmdet.apis`. These apis are '
                      'required in this script! ')
try:
    from mmpose.apis import inference_top_down_pose_model, init_pose_model
except (ImportError, ModuleNotFoundError):
    raise ImportError('Failed to import `inference_top_down_pose_model` and '
                      '`init_pose_model` form `mmpose.apis`. These apis are '
                      'required in this script! ')

#mmdet_root = ''
#mmpose_root = ''
mmdet_root = '/home/Project/Action/mmdetection'
mmpose_root = '/home/Project/Action/mmpose'

args = abc.abstractproperty()
args.det_config = f'{mmdet_root}/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco-person.py'  # noqa: E501
args.det_checkpoint = './faster_rcnn_r50_fpn_1x_coco-person_20201216_175929-d022e227.pth'  # noqa: E501
args.pose_config = f'{mmpose_root}/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w32_coco_256x192.py'  # noqa: E501
args.pose_checkpoint = './hrnet_w32_coco_256x192-c78dce93_20200708.pth'  # noqa: E501

# 使用COCOWholeBody的相关模型
args.pose_config = f'{mmpose_root}/configs/wholebody/2d_kpt_sview_rgb_img/topdown_heatmap/coco-wholebody/hrnet_w48_coco_wholebody_384x288_dark_plus.py'  # noqa: E501
args.pose_checkpoint = './hrnet_w48_coco_wholebody_384x288_dark-f5726563_20200918.pth'  # noqa: E501

args.TargetKeypoints = [0, 6, 5, 12, 11, 14, 13, 16, 15, 20, 17, 8, 10, 7, 9, 116, 132, 95, 111]
args.Chains = [
    [0, 1], [0, 2], [1, 2], [1, 3], [3, 4], [2, 4], [3, 5],
    [5, 7], [7, 9], [4, 6], [6, 8], [8, 10],
    [1, 11], [11, 12], [12, 15], [12, 16],
    [2, 13], [13, 14], [14, 17], [14, 18]
]

def visual_sample(skeletondata, chains, path='visual.jpg'):
    axmax, aymax = skeletondata.max(axis=(0, 1))
    axmin, aymin = skeletondata.min(axis=(0, 1))
    cx, cy = int((axmax + axmin) * 0.5), int((aymax + aymin) * 0.5)
    r = floor(max(aymax - aymin, axmax - axmin) * 0.5 + 0.5)
    axmin, aymin, axmax, aymax = cx - r, cy - r, cx + r, cy + r
    frames = []
    for i, points in enumerate(skeletondata):
        fig = plt.figure(8)
        ax = plt.gca()
        plt.axis([axmin, axmax, aymin, aymax])
        of = GV.gif_frame(fig, ax, chains, points)
        frames.append(of)
    gif.save(frames, path, duration=50)
    pass

def ntu_pose_extraction(args, reflect_map):
    options = args
    target_point = [3, 4, 8, 12, 16, 13, 17, 14, 18, 15, 19, 5, 6, 9, 10, 22, 21, 24, 23]
    chains = [
        [15, 12, 11, 1, 2, 13, 14, 17],
        [0, 1, 3, 5, 7, 9],
        [0, 2, 4, 6, 8, 10],
        [3, 4],
        [12, 16],
        [14, 18]
    ]
    data = NTURGBD.NTURGBDData(
        ntu_root=r'/data/caojihua/data/NTU-RGBD/',
        use_rate=1.0,
        action_ids=['099', '026', '027'],
        camera_ids=None,
        remain_strategy=NTURGBD.NTURGBDData.RemainStrategy.Drop,
        data_type=NTURGBD.NTURGBDData.DataType.Skeleton
    )
    root_node = pAug.AugNode(pAug.AugFuncNoOp())
    Original = root_node.add_child(pAug.AugNode(pAug.AugFuncNoOp()))
    root_node.freeze_node()
    data.set_aug_node_root(root_node, [1.0])
    extract = list()

    for aindex, a in enumerate(data):
        if not options.Visual and not options.Extract:
            print('do nothing')
            sys.exit(1)

        if len(a[1]['nbodys']) == 0:
            continue
        skeletondata = a[1]['skel_body0'][:, :, 0: 2]
        if options.UseFrame <= 0:
            frame = skeletondata.shape[0]
            pass
        else:
            frame = options.UseFrame
        skeletondata = skeletondata[:, target_point, :]
        skeletondata = skeletondata[0: frame]

        if a[0]['action_id'] in reflect_map.keys():
            for sd in skeletondata:
                anno = dict()
                anno['keypoint'] = np.expand_dims(np.expand_dims(sd[..., :2], axis=0), axis=0)
                #anno['keypoint_score'] = np.ones([1, 1, sd.shape[0]], dtype=np.float32)
                anno['keypoint_score'] = np.array([args.SetKeypointScore] * sd[..., 2].size, dtype=np.float32).reshape(sd[..., 2].shape) if args.SetKeypointScore >= 0. else sd[..., 2]
                anno['total_frames'] = len(sd.shape[1])
                anno['label'] = reflect_map[a[0]['action_id']]
                extract.append(anno)
                pass
            pass
        pass
    pass

def image_pose_extraction(args, reflect_map):
    frame_paths = [os.path.join(args.ImageRoot, i) for i in os.listdir(args.ImageRoot)]

    det_model = init_detector(args.det_config, args.det_checkpoint, args.device)
    model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(int(len(frame_paths) / args.BatchSize + 0.5))

    batch_index = 0
    annos = list()
    while batch_index * args.BatchSize < len(frame_paths):
        si, se = batch_index * args.BatchSize, (batch_index + 1) * args.BatchSize
        if args.BaseVocDet:
            _det_results = list()
            for fp in frame_paths[si: se]:
                fp = frame_paths[i]
                voc_id = VOCToolSet.image2id(fp)
                xmldir = os.path.join(args.VOCXmlRoot, VOCToolSet.id2xml(voc_id))
                object_cells = VOCToolSet.extract_object(xmldir)
                img_info = VOCToolSet.get_image_info(xmldir)
                _det_results.append([{'bbox': [oc['bbox']['xmin'], oc['bbox']['ymin'], oc['bbox']['xmax'], oc['bbox']['ymax'], 1.0], 'onehot': reflect_map[oc['name']]} for oc in object_cells if oc['name'] in reflect_map.keys()])
        else:
            _temp_det_results = inference_detector(det_model, frame_paths[si: se])
            _det_results = [[{'bbox': _dr, 'onehot': reflect_map['person']} for _dr in tdr[0] if _dr[4] >= args.DetScoreConf] for tdr in _temp_det_results]
        #det_results = [[{'bbox': _dr, 'onehot': reflect_map['person']} for _dr in drs[0] if _dr[4] >= args.DetScoreConf] for drs in _det_results]
        for dindex, drs in enumerate(_det_results):
            #det_results = [{'bbox': _dr, 'onehot': reflect_map['person']} for _dr in drs[0] if _dr[4] >= args.DetScoreConf]
            det_results = drs
            pose_result = inference_top_down_pose_model(model, frame_paths[si: se][dindex], det_results, format='xyxy')[0]
            poses = list()
            for j, item in enumerate(pose_result):
                kp = np.zeros((1, 1, len(args.TargetKeypoints), 3), dtype=np.float32)
                kp[0, 0] = item['keypoints'][args.TargetKeypoints, :]
                kp[0, 0, :, 1] = -kp[0, 0, :, 1]
                anno = dict()
                anno['keypoint'] = kp[..., :2]
                anno['keypoint_score'] = np.array([args.SetKeypointScore] * kp[..., 2].size, dtype=np.float32).reshape(kp[..., 2].shape) if args.SetKeypointScore >= 0. else kp[..., 2]
                anno['total_frames'] = kp.shape[1]
                anno['label'] = det_results[j]['onehot']
                poses.append(anno)
            annos += poses
            visual_sample(np.concatenate([pose['keypoint'][:, 0] for pose in poses], axis=0), args.Chains, os.path.join(os.path.dirname(args.Output), '{0}-visual-{1}.gif'.format(os.path.basename(args.Output), batch_index * args.BatchSize + dindex))) if batch_index * args.BatchSize + dindex in args.VisualSampleClip and args.DoVisualSample else None
        prog_bar.update()
        batch_index += 1
        pass
    return annos


def voc_pose_extraction(args, reflect_map, skip_postproc=False):
    # todo: 这里读取图像需要更改，改成从VOC数据集中获取单帧图像的det_results(frame_num, person_num, 5), pose_result(1, frame_num, 133, 3)
    frame_paths = [os.path.join(args.ImageRoot, i) for i in os.listdir(args.ImageRoot)]

    model = init_pose_model(args.pose_config, args.pose_checkpoint, args.device)
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))

    annos = list()
    for fpindex, fp in enumerate(frame_paths):
    #for fp in frame_paths[0: 1]:
        voc_id = VOCToolSet.image2id(fp)
        xmldir = os.path.join(args.VOCXmlRoot, VOCToolSet.id2xml(voc_id))
        object_cells = VOCToolSet.extract_object(xmldir)
        img_info = VOCToolSet.get_image_info(xmldir)
        det_results = [{'bbox': [oc['bbox']['xmin'], oc['bbox']['ymin'], oc['bbox']['xmax'], oc['bbox']['ymax'], 1.0], 'onehot': reflect_map[oc['name']]} for oc in object_cells if oc['name'] in reflect_map.keys()]
        # Align input format
        pose_result = inference_top_down_pose_model(model, fp, det_results, format='xyxy')[0]
        poses = list()
        for j, item in enumerate(pose_result):
            # important: 取子集
            kp = np.zeros((1, 1, len(args.TargetKeypoints), 3), dtype=np.float32)
            kp[0, 0] = item['keypoints'][args.TargetKeypoints, :]
            kp[0, 0, :, 1] = -kp[0, 0, :, 1]
            anno = dict()
            anno['keypoint'] = kp[..., :2]
            anno['keypoint_score'] = np.array([args.SetKeypointScore] * kp[..., 2].size, dtype=np.float32).reshape(kp[..., 2].shape) if args.SetKeypointScore >= 0. else kp[..., 2]
            #anno['frame_dir'] = args.VOCImageRoot
            #anno['img_shape'] = (img_info['height'], img_info['width'])
            #anno['original_shape'] = (img_info['height'], img_info['width'])
            #anno['image_name'] = os.path.basename(fp)
            #anno['bbox'] = det_results[j]
            anno['total_frames'] = kp.shape[1]
            # todo: 这里判定类别的方法需要变更：使用VOC的标记，判定skeleton是否位于bbox中，进而指定类别，需要有优先排序，因为目标遮挡可能造成bbox包含多个skeleton实例
            anno['label'] = det_results[j]['onehot']
            poses.append(anno)
        annos += poses
        visual_sample(poses, os.path.join(os.path.dirname(args.Output), '{0}-visual-{1}.jpg'.format(os.path.basename(args.Output), fpindex))) if fpindex in args.VisualSampleClip and args.DoVisualSample else None
        prog_bar.update()
        pass
    return annos


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Pose Annotation for a single NTURGB-D video')
    parser.add_argument('--output', dest='Output', type=str, help='output pickle name')

    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--batch_size', dest='BatchSize', type=int, default=32, action='store', help='指定推理使用的batch size')
    parser.add_argument('--skip-postproc', action='store_true')

    parser.add_argument('--det_score_conf', dest='DetScoreConf', type=float, default=0.7, action='store', help='指定目标检测阈值')

    parser.add_argument('--do_visual_sample', dest='DoVisualSample', action='store_true', help='若指定，则可视化第一个样本')
    parser.add_argument('--visual_sample_clip', dest='VisualSampleClip', type=int, default=[], nargs='+', help='指定进行可视化的帧索引')

    # 针对ntu数据集进行提取
    parser.add_argument('--do_ntu_extraction', dest='DoNTUExtraction', action='store_true', help='提取ntu骨骼数据')
    parser.add_argument('--ntu_root', dest='NTURoot', type=str, action='store', default='', help='指定NTU数据集根目录')

    # 针对视频进行提取
    parser.add_argument('--video', dest='Video', type=str, help='对视频进行提取时指定')

    parser.add_argument('--use_frame', dest='UseFrame', action='store', type=int, default=-1, help='对于单元文件为序列帧(比如视频)指定进行处理的帧数量')

    # 针对图像集进行提取
    parser.add_argument('--image_root', dest='ImageRoot', type=str, default='', action='store', help='指定图像路径')
    # :增加接入VOC标记数据的相关参数，使用图像提取时可以指定voc标记
    parser.add_argument('--base_voc_det', dest='BaseVocDet', action='store_true', help='若指定，则使用voc标注信息(标注框作为目标框，标注类别作为过滤器)')
    parser.add_argument('--voc_xml_root', dest='VOCXmlRoot', type=str, default='', action='store', help='指定VOC标记数据路径')

    parser.add_argument('--target_class', dest='TargetClass', nargs='+', default=[], help='指定voc中标注集中的目标类别集合')
    parser.add_argument('--onehot', dest='Onehot', type=int, nargs='+', default=[], help='指定TargetClass对应的独热码')

    parser.add_argument('--set_keypoint_score', dest='SetKeypointScore', type=float, default=-1.0, action='store', help='指定keypoint的额score，当为负时，使用推理score')


    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import ptvsd
    #ptvsd.enable_attach(address=('127.0.0.1', 65533), redirect_output=True)
    #ptvsd.wait_for_attach()
    pass

    global_args = parse_args()
    args.Output = global_args.Output

    args.device = global_args.device
    args.BatchSize = global_args.BatchSize
    args.skip_postproc = global_args.skip_postproc
    args.DetScoreConf = global_args.DetScoreConf

    args.DoVisualSample = global_args.DoVisualSample
    args.VisualSampleClip = global_args.VisualSampleClip

    args.DoNTUExtraction = global_args.DoNTUExtraction
    args.NTURoot = global_args.NTURoot

    args.video = global_args.Video

    args.ImageRoot = global_args.ImageRoot
    args.TargetClass = global_args.TargetClass
    args.Onehot = global_args.Onehot

    args.SetKeypointScore = global_args.SetKeypointScore

    args.BaseVocDet = global_args.BaseVocDet
    args.VOCXmlRoot = global_args.VOCXmlRoot
    #args.VOCImageRoot = '/data/caojihua/data/0109奔跑/JPEGImages/'
    #args.VOCXmlRoot = '/data/caojihua/data/0109奔跑/AnnotationsCombineYolov5v70Large-default-ForSport/'
    #args.VOCImageRoot = '/data/caojihua/data/0109跳跃/JPEGImages/'
    #args.VOCXmlRoot = '/data/caojihua/data/0109跳跃/AnnotationsCombineYolov5v70Large-default-ForSport/'
    # todo: 解析TargetClass与OneHot
    assert len(args.TargetClass) == len(args.Onehot)
    if len(args.TargetClass) == 0:
        print('no target class selected')
        sys.exit(1)
    reflect_map = dict()
    for i, (tc, oh) in enumerate(zip(args.TargetClass, args.Onehot)):
        tcs = tc.split(',')
        for t in tcs:
            reflect_map[t] = oh
            pass
        pass
    anno = image_pose_extraction(args, reflect_map)

    mmcv.dump(anno, args.Output)
