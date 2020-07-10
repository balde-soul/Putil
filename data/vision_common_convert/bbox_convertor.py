# coding=utf-8
import cv2
import numpy as np
import Putil.base.logger as plog
import Putil.data.convert_to_input as convert_to_input
import Putil.function.gaussian as Gaussion
from enum import Enum

bbox_convertor_logger = plog.PutilLogConfig('bbox_convertor').logger()
bbox_convertor_logger.setLevel(plog.DEBUG)
BBoxConvertToCenterBoxLogger = bbox_convertor_logger.getChild('BBoxConvertToCenterBox')
BBoxConvertToCenterBoxLogger.setLevel(plog.DEBUG)

class BBoxToBBoxTranslator:
    '''
     @brief
     @note
     translate the bbox from original format to the target format
    '''
    class BBoxFormat(Enum):
        '''
         @brief the Format
         @note the supported bbox format
        '''
        # the [left_top_row_index, left_top_col_index, width, height]
        LTWHRC = 0
        # the [left_top_col_index, left_top_row_index, width, height]
        LTWHCR = 1
        #LTWHCR = 2
        #LTWHCR = 1
        pass

    def __init__(self, bbox_in_format, bbox_ret_format):
        self._bbox_in_format = bbox_in_format
        self._bbox_ret_format = bbox_ret_format

        self._translate_func = lambda bbox_input: bbox_input\
            if self._bbox_in_format == self._bbox_ret_format else self._generate_translate_func()
        pass

    def _generate_translate_func(self):
        raise NotImplementedError("this function is not implemented")

    def __call__(self, *args):
        return self._translate_func(*args)
    pass

class BBoxConvertToInputMethod(convert_to_input.ConvertToInput):
    def __init__(self, format):
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
        class_amount,
        input_bbox_format=BBoxToBBoxTranslator.BBoxFormat.LTWHCR,
        sigma=None,
        mu=None,
        resolution=0.05):
        '''
         @brief
         @note
         @param[in] sample_rate
         @param[in] input_bbox_format default: BBoxToBBox.BBoxFormat.LTWHCR
         @param[in] sigma default: [[0.5, 0.0], [0.0, 0.5]]
         @param[in] mu default: [[0.0], [0.0]]
         @param[in] resolution default: 0.05
        '''
        convert_to_input.ConvertToInput.__init__(self)
        self._sample_rate = sample_rate
        self._class_amount = class_amount
        self._weight_func = Gaussion.Gaussian()
        self._weight_func.set_Mu(mu if mu is not None else [[0.0], [0.0]])
        self._weight_func.set_Sigma(sigma if sigma is not None else [[0.1, 0.0], [0.0, 0.1]])

        self._resolution = resolution

        self._format_translator = BBoxToBBoxTranslator(input_bbox_format, BBoxToBBoxTranslator.BBoxFormat.LTWHCR)
        pass

    def __call__(self, *args):
        '''
         @brief generate the box_label from the input
         @param[in] args
         [image, bboxes]
         bboxes: [[top_left_col_i, top_left_row_i, width, height], ...]
         @ret 
         [image, center_box_label_with_weight]
        '''
        image = args[0]
        boxes = args[1]
        classes = args[2]
        box_label = np.zeros(shape=[5, image.shape[0] // self._sample_rate, image.shape[1] // self._sample_rate], \
            dtype=np.float32)
        class_label = np.zeros(shape=[image.shape[0] // self._sample_rate, image.shape[1] // self._sample_rate], \
            dtype=np.float32)
        radiance_factor = np.zeros(shape=[image.shape[0] // self._sample_rate, image.shape[1] // self._sample_rate], \
            dtype=np.float32)
        for box_iter, class_iter in zip(boxes, classes): 
            box = self._format_translator(box_iter)
            x_cell_index = (box[0] + 0.5) // self._sample_rate
            y_cell_index = (box[1] + 0.5) // self._sample_rate

            x_cell_shift = (box[0] + 0.5) % self._sample_rate
            y_cell_shift = (box[1] + 0.5) % self._sample_rate
            w_cell = box[2] / self._sample_rate
            h_cell = box[3] / self._sample_rate

            xregion = [max(round(box[0] - box[2] * 0.5), 0), min(round(box[0] + box[2] * 0.5), image.shape[1])]
            yregion = [max(round(box[1] - box[3] * 0.5), 0), min(round(box[1] + box[3] * 0.5), image.shape[1])]
            #xregion = [round(box[0] - box[2] * 0.5), round(box[0] + box[2] * 0.5)]
            #yregion = [round(box[1] - box[3] * 0.5), round(box[1] + box[3] * 0.5)]
            xamount = xregion[1] - xregion[0]
            x = np.linspace(-1, 1, num=xamount)
            yamount = yregion[1] - yregion[0]
            y = np.linspace(-1, 1, num=yamount)
            x, y = np.meshgrid(x, y)
            shape = x.shape
            coor = np.reshape(np.stack([x, y], axis=-1), [-1, 2])
            weights = self._weight_func(coor).astype(np.float32)
            weights = np.reshape(weights, shape)
            weights = np.pad(weights, ((yregion[0], image.shape[0] - yregion[1]), (xregion[0], image.shape[1] - xregion[1])), mode=lambda vector, iaxis_pad_width, iaxis, kwargs: 0)
            weights = cv2.resize(weights, radiance_factor.shape, interpolation=cv2.INTER_LINEAR)

            box_label[0, int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = 1.0
            box_label[1: 5, int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = [x_cell_shift, y_cell_shift, w_cell, h_cell]

            radiance_factor = np.max(np.stack([radiance_factor, weights], axis=0), axis=0)

            class_label[int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = class_iter

        radiance_factor = (radiance_factor - np.min(radiance_factor)) / (np.max(radiance_factor) - np.min(radiance_factor))
        return image, box_label, class_label, radiance_factor
    pass