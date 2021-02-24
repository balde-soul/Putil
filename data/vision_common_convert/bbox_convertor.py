# coding=utf-8
from importlib import reload
import copy
import cv2
import numpy as np
import Putil.base.logger as plog
import Putil.data.convert_to_input as convert_to_input
reload(convert_to_input)
IOConvertor = convert_to_input.IOConvertor
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
        # 
        LTRBRC = 2
        LTRBCR = 3
        pass

    def __init__(self, bbox_in_format, bbox_ret_format):
        self._bbox_in_format = bbox_in_format
        self._bbox_ret_format = bbox_ret_format

        if self._bbox_in_format == self._bbox_ret_format:
            self._translate_func = self._directed
        else:
            self._translate_func = self._generate_translate_func
            pass
        pass

    def ToLTRB(self):
        pass

    def _directed(self, box):
        return box

    def _generate_translate_func(self):
        raise NotImplementedError("this function is not implemented")

    def __call__(self, *args):
        return self._translate_func(*args)
    pass

class BBoxConvertToInputMethod(IOConvertor):
    def __init__(self, format):
        '''
        '''
        IOConvertor.__init__(self, IOConvertor.IODirection.Unknow)
        pass

    def __call__(self, *args):
        return args
    pass


class CenterNerIOConvertor(IOConvertor):
    def __init__(
        self, 
        sample_rate, 
        class_amount,
        io,
        input_bbox_format=BBoxToBBoxTranslator.BBoxFormat.LTWHCR,
        sigma=None,
        mu=None,
        resolution=0.05,
        **kwargs):
        '''
         @brief
         fit the output from the Data with the CenterNet 
         @note
         @param[in] sample_rate
         the downsample rate of the CenterNet, the label
         @param[in] class_amount
         @param[in] io
         take a look at IOConvertor.__init__.param{io}
         @param[in] input_bbox_format default: BBoxToBBox.BBoxFormat.LTWHCR
         @param[in] sigma default: [[0.5, 0.0], [0.0, 0.5]]
         @param[in] mu default: [[0.0], [0.0]]
         @param[in] resolution default: 0.05
        '''
        IOConvertor.__init__(self, io)
        self._sample_rate = sample_rate
        self._class_amount = class_amount
        self._weight_func = Gaussion.Gaussian()
        self._weight_func.set_Mu(mu if mu is not None else [[0.0], [0.0]])
        self._weight_func.set_Sigma(sigma if sigma is not None else [[0.1, 0.0], [0.0, 0.1]])

        self._resolution = resolution

        self._format_translator = BBoxToBBoxTranslator(input_bbox_format, BBoxToBBoxTranslator.BBoxFormat.LTWHCR)

        assert self._io != IOConvertor.IODirection.Unknow
        pass

    def __call__(self, *args):
        '''
         @brief generate the box_label from the input
         @param[in] args
         [0] image[IOConvertor.IODirection.InputConvertor]
         image with shape [height, width, channel] in numpy
         [1] boxes
         [[top_left_col_i, top_left_row_i, width, height], ...]
         [2] classes 
         [[class], ...]
         [image, bboxes]
         bboxes: [[obj_or_not, top_left_col_i, top_left_row_i, width, height], ...]
         @ret 
         [0]
        '''
        if self._io == IOConvertor.IODirection.InputConvertion:
            image = args[0]
            boxes = args[1]
            classes = args[2]
            out_height = image.shape[0] // self._sample_rate
            out_width = image.shape[1] // self._sample_rate
            obj_label = np.zeros(shape=[1, out_height, out_width], \
                dtype=np.float32)
            box_label = np.zeros(shape=[4, out_height, out_width], \
                dtype=np.float32)
            class_label = np.zeros(shape=[out_height, out_width], \
                dtype=np.int64)
            radiance_factor = np.zeros(shape=[out_height, out_width], \
                dtype=np.float32)
            for box_iter, class_iter in zip(boxes, classes): 
                box = self._format_translator(box_iter)
                x_cell_index = (box[0]) // self._sample_rate
                y_cell_index = (box[1]) // self._sample_rate

                standard_cell_center_x = x_cell_index + 1.5
                standard_cell_center_y = y_cell_index + 1.5
                x_cell_shift = (box[0] - standard_cell_center_x) / self._sample_rate
                y_cell_shift = (box[1] - standard_cell_center_y) / self._sample_rate
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

                obj_label[0, int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = 1.0
                box_label[:, int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = [x_cell_shift, y_cell_shift, w_cell, h_cell]

                radiance_factor = np.max(np.stack([radiance_factor, weights], axis=0), axis=0)

                class_label[int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = class_iter

            radiance_factor = (radiance_factor - np.min(radiance_factor)) / (np.max(radiance_factor) - np.min(radiance_factor) + 1e-32)
            return image, box_label, class_label, obj_label, radiance_factor
        else:
            images = args[0]
            boxes = args[1]
            classes = args[2]
            objs = args[3]
            objs = np.round(objs)

            general_boxes = list()
            general_classes = list()
            standard_cell_center_x_y_s = list()

            for _index, (box_out, class_out, obj_out, image) in enumerate(zip(boxes, classes, objs, images)):
                indexs = np.where(obj_out == 1)
                index_list = [np.expand_dims(i, -1) for i in indexs]
                index_list[0] = np.meshgrid(list(range(0, class_out.shape[0])), index_list[0])[0]
                standard_center_x_y = np.concatenate(index_list[1: ], -1) * self._sample_rate + 1.5
                index_list = [np.expand_dims(i, -1) for i in indexs]
                index_list[0] = np.meshgrid(list(range(0, class_out.shape[0])), index_list[0])[0]
                general_class = np.argmax(class_out[index_list], axis=-1)
                index_list = [np.expand_dims(i, -1) for i in indexs]
                index_list[0] = np.meshgrid(list(range(0, box_out.shape[0])), index_list[0])[0]
                general_box = box_out[index_list]
                general_box[:, 2: 4] = general_box[:, 2: 4] * self._sample_rate
                general_box[:, 0: 2] = general_box[:, 0: 2] * self._sample_rate + standard_center_x_y - general_box[:, 2: 4] * 0.5
                # remove the box whose top-left is out of the image
                # remove the box whose bottom-right is out of the image
                top_left_index = np.where(general_box[:, 0: 2] > 0, True, False)
                top_left_index = top_left_index[:, 0] * top_left_index[:, 1]
                bottom_right_index = np.where((general_box[:, 0: 2] + general_box[:, 2:]) > [image.shape[-2], image.shape[1]], True, False)
                bottom_right_index = bottom_right_index[:, 0] * bottom_right_index[:, 1]
                accept_index = top_left_index * bottom_right_index
                general_box = general_box[accept_index, :]
                general_class = general_class[accept_index]
                standard_center_x_y = standard_center_x_y[accept_index, :]
                general_boxes.append(general_box.tolist())
                general_classes.append(general_class.tolist())
                standard_cell_center_x_y_s.append(standard_center_x_y.tolist())
            return images, general_boxes, general_classes, standard_cell_center_x_y_s
        pass
    pass

BBoxConvertToCenterBox = CenterNerIOConvertor


class CenterNerIOConvertor(IOConvertor):
    def __init__(
        self, 
        sample_rate, 
        class_amount,
        io,
        input_bbox_format=BBoxToBBoxTranslator.BBoxFormat.LTWHCR,
        sigma=None,
        mu=None,
        resolution=0.05,
        **kwargs):
        '''
         @brief
         fit the output from the Data with the CenterNet 
         @note
         @param[in] sample_rate
         the downsample rate of the CenterNet, the label
         @param[in] class_amount
         @param[in] io
         take a look at IOConvertor.__init__.param{io}
         @param[in] input_bbox_format default: BBoxToBBox.BBoxFormat.LTWHCR
         @param[in] sigma default: [[0.5, 0.0], [0.0, 0.5]]
         @param[in] mu default: [[0.0], [0.0]]
         @param[in] resolution default: 0.05
        '''
        IOConvertor.__init__(self, io)
        self._sample_rate = sample_rate
        self._class_amount = class_amount
        self._weight_func = Gaussion.Gaussian()
        self._weight_func.set_Mu(mu if mu is not None else [[0.0], [0.0]])
        self._weight_func.set_Sigma(sigma if sigma is not None else [[0.1, 0.0], [0.0, 0.1]])

        self._resolution = resolution

        self._format_translator = BBoxToBBoxTranslator(input_bbox_format, BBoxToBBoxTranslator.BBoxFormat.LTWHCR)

        assert self._io != IOConvertor.IODirection.Unknow
        pass

    def __call__(self, *args):
        '''
         @brief generate the box_label from the input
         @param[in] args
         [0] image[IOConvertor.IODirection.InputConvertor]
         image with shape [height, width, channel] in numpy
         [1] boxes
         [[top_left_col_i, top_left_row_i, width, height], ...]
         [2] classes 
         [[class], ...]
         [image, bboxes]
         bboxes: [[obj_or_not, top_left_col_i, top_left_row_i, width, height], ...]
         @ret 
         [0]
        '''
        if self._io == IOConvertor.IODirection.InputConvertion:
            image = args[0]
            boxes = args[1]
            classes = args[2]
            out_height = image.shape[0] // self._sample_rate
            out_width = image.shape[1] // self._sample_rate
            obj_label = np.zeros(shape=[1, out_height, out_width], \
                dtype=np.float32)
            box_label = np.zeros(shape=[4, out_height, out_width], \
                dtype=np.float32)
            class_label = np.zeros(shape=[out_height, out_width], \
                dtype=np.int64)
            radiance_factor = np.zeros(shape=[out_height, out_width], \
                dtype=np.float32)
            for box_iter, class_iter in zip(boxes, classes): 
                box = self._format_translator(box_iter)
                x_cell_index = (box[0]) // self._sample_rate
                y_cell_index = (box[1]) // self._sample_rate

                standard_cell_center_x = x_cell_index + 1.5
                standard_cell_center_y = y_cell_index + 1.5
                x_cell_shift = (box[0] - standard_cell_center_x) / self._sample_rate
                y_cell_shift = (box[1] - standard_cell_center_y) / self._sample_rate
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

                obj_label[0, int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = 1.0
                box_label[:, int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = [x_cell_shift, y_cell_shift, w_cell, h_cell]

                radiance_factor = np.max(np.stack([radiance_factor, weights], axis=0), axis=0)

                class_label[int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = class_iter

            radiance_factor = (radiance_factor - np.min(radiance_factor)) / (np.max(radiance_factor) - np.min(radiance_factor) + 1e-32)
            return image, box_label, class_label, obj_label, radiance_factor
        else:
            images = args[0]
            boxes = args[1]
            classes = args[2]
            objs = args[3]
            objs = np.round(objs)

            general_boxes = list()
            general_classes = list()
            standard_cell_center_x_y_s = list()

            for _index, (box_out, class_out, obj_out, image) in enumerate(zip(boxes, classes, objs, images)):
                indexs = np.where(obj_out == 1)
                index_list = [np.expand_dims(i, -1) for i in indexs]
                index_list[0] = np.meshgrid(list(range(0, class_out.shape[0])), index_list[0])[0]
                standard_center_x_y = np.concatenate(index_list[1: ], -1) * self._sample_rate + 1.5
                index_list = [np.expand_dims(i, -1) for i in indexs]
                index_list[0] = np.meshgrid(list(range(0, class_out.shape[0])), index_list[0])[0]
                general_class = np.argmax(class_out[index_list], axis=-1)
                index_list = [np.expand_dims(i, -1) for i in indexs]
                index_list[0] = np.meshgrid(list(range(0, box_out.shape[0])), index_list[0])[0]
                general_box = box_out[index_list]
                general_box[:, 2: 4] = general_box[:, 2: 4] * self._sample_rate
                general_box[:, 0: 2] = general_box[:, 0: 2] * self._sample_rate + standard_center_x_y - general_box[:, 2: 4] * 0.5
                # remove the box whose top-left is out of the image
                # remove the box whose bottom-right is out of the image
                top_left_index = np.where(general_box[:, 0: 2] > 0, True, False)
                top_left_index = top_left_index[:, 0] * top_left_index[:, 1]
                bottom_right_index = np.where((general_box[:, 0: 2] + general_box[:, 2:]) > [image.shape[-2], image.shape[1]], True, False)
                bottom_right_index = bottom_right_index[:, 0] * bottom_right_index[:, 1]
                accept_index = top_left_index * bottom_right_index
                general_box = general_box[accept_index, :]
                general_class = general_class[accept_index]
                standard_center_x_y = standard_center_x_y[accept_index, :]
                general_boxes.append(general_box.tolist())
                general_classes.append(general_class.tolist())
                standard_cell_center_x_y_s.append(standard_center_x_y.tolist())
            return images, general_boxes, general_classes, standard_cell_center_x_y_s
        pass
    pass


BBoxConvertToCenterBox = CenterNerIOConvertor


from Putil.data.io_convertor_with_torch_module import IOConvertorModule
import torch 
from torch.nn import Module


class CenterNetDecode(Module):
    def __init__(self):
        Module.__init__(self)

    def forward(self, pre_box, pre_class, pre_obj):
        pass