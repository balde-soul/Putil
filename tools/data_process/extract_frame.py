# coding=utf-8
import os, random, cv2, sys
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Putil.base.arg_base import DictAction
import Putil.data.aug as pAug
random.seed(1995)

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', dest='VideoPath', type=str, default='', action='store', help='指定抽帧视频文件')
parser.add_argument('--save_to', dest='SaveTo', type=str, default='', action='store', help='指定图像保存目录')
parser.add_argument('--fps', dest='FPS', type=float, default=1, action='store', help='指定抽帧fps,默认为1,每秒抽一帧')
#parser.add_argument('--roi', dest='ROI', nargs='+', action=DictAction, default={}, \
#    help='以多边形的格式设置感兴趣区域,格式: --roi [(x1, y1), (x2, y2), ...](x\in [0, +\infty), y\in [0, +\infty), 图像有长宽限制-1代表取最大值)')
#parser.add_argument('--roi_mode', dest='ROIMode', type=str, action='store', default='Keep', help='指定感兴趣区域过滤的模式\n\
#    Keep: 保持原始尺寸，ROI区域之外的置黑色\
#    Crop: 截取ROI的最大外接水平矩形部分(未支持)')
options = parser.parse_args()

#class ROIProcess:
#    def __init__(self):
#        pass
#
#    def __call__(self, *args):
#        pass
#    pass
#
#process_nodes = pAug.AugNode(ROIProcess(parser.ROI, parser.ROIMode))
#process_nodes.freeze_node()


def select_frame(video, fps, save_dir, ProcessNodes=None):
    # select frames of video contained traffic lights, save frames selected into save_dir
    cap = cv2.VideoCapture(video)
    video_name = os.path.basename(video)

    video_FPS = cap.get(cv2.CAP_PROP_FPS)
    # print(video_FPS)
    gap_frame = int(video_FPS / fps)

    frame_cnt = 0
    num = 0

    while (True):
        success, frame = cap.read()
        if not success:
            break

        if frame_cnt % gap_frame == 0:
            # save fram into save_dir
            path = os.path.join(save_dir, video_name.split('.')[0] + '_' + str(num) + '.jpg')
            cv2.imwrite(path, frame)
            num += 1
            pass

        frame_cnt += 1
    cap.release()
    # print('select video %s finished!' % video_name)
    pass

if not os.path.exists(options.SaveTo):
    os.mkdir(options.SaveTo)
    pass

if not os.path.exists(options.VideoPath):
    print('video {0} does not exist'.format(options.VideoPath))
    pass

select_frame(options.VideoPath, fps=options.FPS, save_dir=options.SaveTo)