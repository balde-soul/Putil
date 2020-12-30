#In[]
# coding=utf-8

import os
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from Putil.data.vision_data_aug.image_aug import Noise
import numpy as np
from matplotlib import pyplot as plt
import cv2
import numpy as np

def contrast(img0):   
    img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY) #彩色转为灰度图片
    m, n = img1.shape
    #图片矩阵向外扩展一个像素
    img1_ext = cv2.copyMakeBorder(img1,1,1,1,1,cv2.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape

    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 + 
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式
    print(cg)

image = cv2.imread('./test/data/vision_data_aug/test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
contrast(image)
plt.imshow(image)
plt.show()
#image = np.zeros(shape=[100, 100, 3], dtype=np.float32)
noise = Noise()
noise.mu = 0
noise.sigma = 40
image_with_noise = noise(image)

plt.imshow(image_with_noise)
contrast(image_with_noise)
plt.show()