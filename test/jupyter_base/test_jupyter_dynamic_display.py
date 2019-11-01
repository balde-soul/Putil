# coding=utf-8
import Putil.jupyter_base.jupyter_dynamic_display as jdd
import cv2
import numpy as np

video_path = target_v_h
video = cv2.VideoCapture()
video.open(video_path)
imgs = []
while True:
    is_valid, img = video.read()
    if not is_valid:
        break
    imgs.append(img[:, :, [2, 1, 0]])
    video.release()
    jdd.plot_sequence_images(imgs)
    pass

# demo of plot_animation_function
frames = 100


def GeneratorXY(frames):
    for i in range(frames):
        x = np.linspace(0, 2, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        yield (x, y)
        pass
    pass


jdd.plot_animation_function(GeneratorXY(frames), frames=frames, figsize=(8, 6), xlim=(0, 2), ylim=(-2, 2))
