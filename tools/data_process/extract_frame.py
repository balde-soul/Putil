# coding=utf-8
import pandas as pd
import os, copy, json, random, sys, cv2
import argparse
random.seed(1995)

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', dest='VideoPath', type=str, default='', action='store', help='指定抽帧视频文件')
parser.add_argument('--save_to', dest='SaveTo', type=str, default='', action='store', help='指定图像保存目录')
parser.add_argument('--fps', dest='FPS', type=int, default=1, action='store', help='指定抽帧fps,默认为1,每秒抽一帧')
options = parser.parse_args()

def select_frame(video, fps=1, save_dir='/media/s4/ae13bf8c-93ea-4157-bb06-1ef3acc87a19/rgb/rgb_test'):
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
            cv2.imwrite(path, frame[:, :, ::-1])
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