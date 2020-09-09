# coding=utf-8
#In[]:
import matplotlib.pyplot as plt
import numpy as np
import cv2


class PointVisual:
    def __init__(self):
        pass

    def visual_on_image(self, image, point_weight, color_map=None):
        '''
         @brief
         @note make point visualization on the image
         @param[in] image [height, width, channel], support one or three channel
         @param[in] point_weight [height, width, point_type_amount], range [0, 1]
         @param[in] color_map [point_type_amount, channel] or list
        '''
        if color_map is None:
            color_map = [[0, 255, 0]] * point_weight.shape[-1]
        if type(color_map).__name__ == 'list':
            color_map = np.array(color_map)
        
        amount = point_weight.shape[-1]
        ret = 0
        for pw, cm in zip(np.transpose(point_weight, [-1, 0, 1]), color_map):
            pw = np.expand_dims(pw, -1)
            front = pw * np.reshape(cm, [1, 1, 3])
            k = (1 - pw) * image + front
            ret = ret +  1 / amount * k
        #plt.imshow(t.astype(np.uint8))
        #plt.show()
        #plt.imshow(tt)
        #plt.show()
        return ret.astype(np.uint8)

    def visual_index(self, image, point_index, color=None):
        if color is None:
            color = [0, 255, 0]
            pass
        point_index = np.array(point_index)
        if point_index.dtype != np.int:
            point_index = point_index.astype(np.int)
            pass
        if point_index.shape == (2, ):
            point_index = np.reshape(point_index, [-1, 2])
            pass
        for point in point_index:
            image = cv2.circle(image, center=tuple(point), radius=1, color=color, thickness=3)
            pass
        return image
        pass


#from PIL import Image
#import numpy as np
#import Putil.sampling.MCMC.metropolis_hasting as pmh
#from Putil.function.gaussian import Gaussian
#
#image = np.array(Image.open('/data2/process_data/caojihua/workspace/Putil/trainer/visual/image_visual/000000123413.jpg'))
#print(image.shape)
#iv = PointVisual()
#
#size = 50
#class_amount = 2
#g = Gaussian()
#g.set_Mu([[0.], [0.]])
#g.set_Sigma([[9., 0.], [0., 9.]])
#index = np.linspace(-0.5, 0.5, size)
#weight = np.zeros(list(image.shape[0: 2]) + [class_amount])
#t = g(np.stack(np.meshgrid(index, index), -1))
#t = (t - np.min(t, axis=(0, 1))) / (np.max(t, axis=(0, 1)) - np.min(t, axis=(0, 1)))
#weight[int(image.shape[0] * 0.5): int(image.shape[0] * 0.5) + size, int(image.shape[1] * 0.5) : int(image.shape[1] * 0.5) + size, 0] = t
#weight[int(image.shape[0] * 0.5) - 2 * size: int(image.shape[0] * 0.5) - size, int(image.shape[1] * 0.5) - size: int(image.shape[1] * 0.5), 1] = t
#weight[int(image.shape[0] * 0.5) + 2 * size: int(image.shape[0] * 0.5) + 3 * size, int(image.shape[1] * 0.5) + size: int(image.shape[1] * 0.5) + 2 * size, 0] = t
#weight[int(image.shape[0] * 0.5) + 2 * size: int(image.shape[0] * 0.5) + 3 * size, int(image.shape[1] * 0.5): int(image.shape[1] * 0.5) + size, 0] = t
#
#a = iv.visual_on_image(image, weight, 
#[[0, 0, 255], [0, 255, 0]])
#plt.imshow(a)
#plt.show()
#b = iv.visual_on_image(image, weight)
#plt.imshow(b)
#plt.show()
#c = iv.visual_index(image, [[20, 20], [100, 100]], [255, 0, 0])
#plt.imshow(c)
#plt.show()