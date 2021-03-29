# coding=utf-8
from Putil.data.io_convertor import Encode
from Putil.np.box import BBoxToBBoxTranslator

##@brief
# fit the output from the Data with the CenterNet 
# @note
# @param[in] sample_rate
# the downsample rate of the CenterNet, the label
# @param[in] class_amount
# @param[in] io
# take a look at convert_to_input.IOConvertor.__init__.param{io}
# @param[in] input_bbox_format default: BBoxToBBox.BBoxFormat.LTWHCR
# @param[in] sigma default: [[0.5, 0.0], [0.0, 0.5]]
# @param[in] mu default: [[0.0], [0.0]]
# @param[in] resolution default: 0.05
class CenterNetIOConvertorOriginal(Encode):
    def __init__(
        self, 
        sample_rate, 
        class_amount,
        input_bbox_format=BBoxToBBoxTranslator.BBoxFormat.LTWHCR,
        sigma=None,
        mu=None,
        resolution=0.05,
        radiantion_range=10,
        **kwargs):
        Encode.__init__(self)
        self._sample_rate = sample_rate
        self._class_amount = class_amount
        self._weight_func = Gaussion.Gaussian()
        self._Mu = mu if mu is not None else [[0.0], [0.0]]
        self._weight_func.set_Mu(self._Mu)
        self._Sigma = sigma if sigma is not None else [[0.0001, 0.0], [0.0, 0.0001]]
        self._weight_func.set_Sigma(self._Sigma)

        self._conf_interval_x_y_low, self._conf_interval_x_y_high = stats.norm.interval(
            0.999, loc=[self._Mu[0][0], self._Mu[1][0]], scale=[self._Sigma[0][0], self._Sigma[1][1]])
        self._conf_interval_x_y_low, self._conf_interval_x_y_high = ([-1.0, -1.0], [1.0, 1.0])

        self._resolution = resolution

        self._format_translator = BBoxToBBoxTranslator(input_bbox_format, BBoxToBBoxTranslator.BBoxFormat.LTWHCR)

        assert self._io != convert_to_input.IOConvertor.IODirection.Unknow and self._io != convert_to_input.IOConvertor.IODirection.OutputConvertion
        pass

    def __call__(self, *args):
        '''
         @brief generate the box_label from the input
         @param[in] args
         [0] image[convert_to_input.IOConvertor.IODirection.InputConvertor]
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
        #import pdb;pdb.set_trace()
        BBoxConvertToCenterBoxLogger.debug('start convert')
        image = args[0]
        image_height, image_width = image.shape[0: 2]
        boxes = args[1]
        base_information = args[2]
        classes = args[-1]
        out_height = image.shape[0] // self._sample_rate
        out_width = image.shape[1] // self._sample_rate
        obj_label = np.zeros(shape=[self._class_amount, out_height, out_width], \
            dtype=np.float32)
        box_label = np.zeros(shape=[4, out_height, out_width], \
            dtype=np.float32)
        class_label = np.ones(shape=[out_height, out_width], \
            dtype=np.int64) * -1
        radiance_factor = np.zeros(shape=[self._class_amount, out_height, out_width], \
            dtype=np.float32)
        #import pdb; pdb.set_trace()
        boxes = np.array(self._format_translator(boxes))
        boxes[:, 0: 2] = boxes[:, 0: 2] + 0.5
        boxes[:, 2: 4] = boxes[:, 0: 2] + boxes[:, 2: 4]
        wh_downsample = (boxes[:, 2: 4] - boxes[:, 0: 2]) / self._sample_rate
        boxes_index = boxes // self._sample_rate
        center_xy = (boxes[:, 0: 2] + boxes[:, 2: 4]) * 0.5
        center_xy_index = center_xy // self._sample_rate
        xy_shift = (center_xy - (center_xy_index + 0.5) * self._sample_rate) / self._sample_rate
        for box_iter, wh_downsample_iter, box_index_iter, class_iter, center_xy_index_iter, xy_shift_iter in zip(boxes, wh_downsample, boxes_index, classes, center_xy_index, xy_shift): 
            x_cell_index = center_xy_index_iter[0]
            y_cell_index = center_xy_index_iter[1]
            x_cell_shift = xy_shift_iter[0]
            y_cell_shift = xy_shift_iter[1]
            box_x1_index = box_index_iter[0]
            box_y1_index = box_index_iter[1]
            box_x2_index = box_index_iter[2]
            box_y2_index = box_index_iter[3]

            half_w = min(x_cell_index - box_x1_index, box_x2_index - x_cell_index)
            xregion = [x_cell_index - half_w, x_cell_index + half_w]
            half_h = min(y_cell_index - box_y1_index, box_y2_index - y_cell_index)
            yregion = [y_cell_index - half_h, y_cell_index + half_h]
            xamount = xregion[1] - xregion[0] + 1
            x = np.linspace(0.0 if xamount == 1 else self._conf_interval_x_y_low[0], 0.0 if xamount == 1 else self._conf_interval_x_y_high[0], 
            num=xamount)
            yamount = yregion[1] - yregion[0] + 1
            y = np.linspace(0.0 if yamount == 1 else self._conf_interval_x_y_low[1], 0.0 if yamount == 1 else self._conf_interval_x_y_high[1],
            num=yamount)
            x, y = np.meshgrid(x, y)
            shape = x.shape
            coor = np.reshape(np.stack([x, y], axis=-1), [-1, 2])
            weights = self._weight_func(coor).astype(np.float32)
            weights = np.reshape(weights, shape)
            #weights = np.pad(weights, ((int(yregion[0]), int(out_height - yregion[1] - 1.0)), (int(xregion[0]), int(out_width - xregion[1] - 1.))),
            #mode=lambda vector, iaxis_pad_width, iaxis, kwargs: 0)
            radiance_factor[class_iter, int(yregion[0]): int(yregion[1] + 1), int(xregion[0]): int(xregion[1] + 1)] = \
                np.max(np.stack([radiance_factor[class_iter, int(yregion[0]): int(yregion[1] + 1), int(xregion[0]): int(xregion[1] + 1)], weights], axis=0), axis=0)

            obj_label[class_iter, int(y_cell_index), int(x_cell_index)] = 1.0
            box_label[:, int(y_cell_index), int(x_cell_index)] = [x_cell_shift, y_cell_shift, wh_downsample_iter[0], wh_downsample_iter[1]]

            class_label[int(y_cell_index + 0.5), int(x_cell_index + 0.5)] = class_iter

        radiance_factor = (radiance_factor - np.min(radiance_factor)) / (np.max(radiance_factor) - np.min(radiance_factor) + 1e-32)
        BBoxConvertToCenterBoxLogger.debug('end convert')
        return image, box_label, class_label, obj_label, base_information, radiance_factor
    pass