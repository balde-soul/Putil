# coding=utf-8
import numpy as np
import Putil.base.logger as plog
import Putil.data.convert_to_input as convert_to_input
import Putil.function.gaussian as Gaussion

bbox_convertor_logger = plog.PutilLogConfig('bbox_convertor').logger()
bbox_convertor_logger.setLevel(plog.DEBUG)
BBoxConvertToCenterBoxLogger = bbox_convertor_logger.getChild('BBoxConvertToCenterBox')
BBoxConvertToCenterBoxLogger.setLevel(plog.DEBUG)

class BBoxConvertToInputMethod(convert_to_input.ConvertToInput):
    def __init__(self, config):
        '''
        '''
        convert_to_input.ConvertToInput.__init__(self)
        pass

    def __call__(self, *args):
        return args
    pass


class BBoxConvertToCenterBox(convert_to_input.ConvertToInput):
    def __init__(
        self, 
        sample_rate, 
        sigma=np.array([[0.5, 0.0], [0.0, 0.5]], dtype=np.float32), 
        mu=np.array([[0.0], [0.0]], dtype=np.float32)):
        convert_to_input.ConvertToInput.__init__(self)
        self._sample_rate = sample_rate
        self._weight_func = Gaussion.Gaussian()
        self._weight_func.set_Mu(mu)
        self._weight_func.set_Sigma(sigma)
        pass

    def __call__(self, *args):
        '''
         @brief generate the label from the input
         @param[in] args
         [image, bboxes]
         @ret 
         [image, center_label_with_weight]
        '''
        image = args[0]
        boxes = args[1]
        label = np.zeros(shape=[image.shape[0] // self._sample_rate, image.shape[1] // self._sample_rate, 4], dtype=np.float32)
        temp_weight = np.zeros(shape=[image.shape[0], image.shape[1], 4], dtype=np.float32) 
        for box in boxes:
            x_cell_index = (box[0] + 0.5) // self._sample_rate
            y_cell_index = (box[1] + 0.5) // self._sample_rate

            x_cell_shift = (box[0] + 0.5) % self._sample_rate
            y_cell_shift = (box[1] + 0.5) % self._sample_rate
            w_cell = box[2] / self._sample_rate
            h_cell = box[3] / self._sample_rate

            xregion = [round(box[0] - box[2] * 0.5), round(box[0] + box[2] * 0.5)]
            yregion = [round(box[1] - box[3] * 0.5), round(box[1] + box[3] * 0.5)]
            x = np.linspace(-(xregion[1] - xregion[0]) * 0.5, (xregion[1] - xregion[0]) * 0.5, num=xregion[1] - xregion[0] + 1)
            y = np.linspace(-(yregion[1] - yregion[0]) * 0.5, (yregion[1] - yregion[0]) * 0.5, num=yregion[1] - yregion[0] + 1)
            coor = np.meshgrid(x, y)
            coor = np.reshape(np.stack(coor, axis=-1), [-1, 2])
            weights = self._weight_func.calc(coor)
            label[y_cell_index][x_cell_index][0] = 1.0
            label[y_cell_index][x_cell_index][1: 5] = [x_cell_shift, y_cell_shift, w_cell, h_cell]
        return args
##In[]
#from PIL import Image
#import numpy as np
#import matplotlib.pyplot as plt
#ia = np.zeros(shape=[50, 100, 3], dtype=np.uint8)
#print(ia.shape)
#plt.imshow(ia)
#plt.show()
#image = Image.fromarray(ia)
#print(image.size)
#print(image.height)
#plt.imshow(image)
#plt.show()
#
#print(np.array(image).shape)
#
#9%4
#
#np.floor(9.8)