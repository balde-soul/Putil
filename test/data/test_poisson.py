# coding=utf-8
#In[]
import cv2
import os
import numpy as np

code_patch_dir = '/data2/process_data/caojihua/data/code_patches/'
background_dir = '/data2/Public_Data/COCO/unzip_data/2017/train2017' 

codes = os.listdir(code_patch_dir)
bgs = os.listdir(background_dir)

#In[]
def read_img(code_patch_dir, background_dir, code_name, bg_name):
    code_path = os.path.join(code_patch_dir, code_name)
    code_img = cv2.imread(code_path, cv2.IMREAD_GRAYSCALE)

    bg_path = os.path.join(background_dir, bg_name)
    bg_img = cv2.imread(bg_path, cv2.IMREAD_GRAYSCALE)
    return code_img, bg_img

code_img, bg_img = read_img(code_patch_dir, background_dir, codes[1], bgs[0])
code_img = cv2.resize(code_img, (bg_img.shape[1] // 4, bg_img.shape[0] // 4))
print(code_img.shape)
print(bg_img.shape)
import matplotlib.pyplot as plt
plt.imshow(code_img, cmap='gray')
plt.show()
plt.imshow(bg_img, cmap='gray')
plt.show()
#In[]
code_img_bgr = cv2.cvtColor(code_img, cv2.COLOR_GRAY2BGR)
bg_img_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
mask = 255 * np.ones(code_img_bgr.shape, code_img_bgr.dtype)
center = (bg_img_bgr.shape[0] // 2 , bg_img_bgr.shape[1] // 2)
result = cv2.seamlessClone(code_img_bgr, bg_img_bgr, mask, center, cv2.NORMAL_CLONE)
plt.imshow(result)
plt.show()
#In[]
code_img_bgr = cv2.cvtColor(code_img, cv2.COLOR_GRAY2BGR)
bg_img_bgr = cv2.cvtColor(bg_img, cv2.COLOR_GRAY2BGR)
code_img_bgr_fill_to_bg = np.zeros(bg_img_bgr.shape, bg_img_bgr.dtype)
h_begin = code_img_bgr.shape[0] // 2 - code_img_bgr.shape[0] // 2
w_begin = code_img_bgr.shape[1] // 2 - code_img_bgr.shape[1] // 2
code_img_bgr_fill_to_bg[h_begin: code_img_bgr.shape[0] + h_begin, w_begin: code_img_bgr.shape[1] + w_begin, :] = code_img_bgr
mask = 255 * np.ones(bg_img_bgr.shape, bg_img_bgr.dtype)
mask[h_begin: code_img_bgr.shape[0] + h_begin, w_begin: code_img_bgr.shape[1] + w_begin, :] = 0
center = (bg_img_bgr.shape[0] // 2 , bg_img_bgr.shape[1] // 2)
result = cv2.seamlessClone(bg_img_bgr, code_img_bgr, mask, center, cv2.NORMAL_CLONE)
plt.imshow(result)
plt.show()
#In[]
print(cv2.seamlessClone.__doc__)