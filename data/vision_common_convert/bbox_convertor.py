# coding=utf-8
import numpy as np
import Putil.base.logger as plog
import Putil.data.convert_to_input as convert_to_input

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
    def __init__(self, sample_rate):
        convert_to_input.ConvertToInput.__init__(self)
        self._sample_rate = sample_rate
        pass

    def __call__(self, *args):
        image = args[0]
        boxes = args[1]
        label = np.zeros(shape=[image.shape[0] // self._sample_rate, image.shape[1] // self._sample_rate, 4], dtype=np.float32)
        for box in boxes:
            x_cell = box[0] % / self._sample_rate
            y_cell = box[1] / self._sample_rate
            w_cell = box[2] / self._sample_rate
            h_cell = box[3] / self._sample_rate
        return args
#In[]
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
ia = np.zeros(shape=[50, 100, 3], dtype=np.uint8)
print(ia.shape)
plt.imshow(ia)
plt.show()
image = Image.fromarray(ia)
print(image.size)
print(image.height)
plt.imshow(image)
plt.show()

print(np.array(image).shape)

9%4