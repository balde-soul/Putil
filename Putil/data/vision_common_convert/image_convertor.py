# coding=utf-8
from enum import Enum
import cv2
import numpy as np
import Putil.base.logger as plog
import Putil.data.convert_to_input as convert_to_input
import Putil.function.gaussian as Gaussion

bbox_convertor_logger = plog.PutilLogConfig('bbox_convertor').logger()
bbox_convertor_logger.setLevel(plog.DEBUG)
BBoxConvertToCenterBoxLogger = bbox_convertor_logger.getChild('BBoxConvertToCenterBox')
BBoxConvertToCenterBoxLogger.setLevel(plog.DEBUG)

class ImageConvertToInputMethod(convert_to_input.ConvertToInput):
    def __init__(self):
        pass

    def __call__(self, *args):
        image = args[0]
        image_id = args[1]
        return image, np.array(image_id)
