# coding = utf-8
import tensorflow as tf
import tensorflow.contrib.layers as layers
from colorama import Fore
import numpy as np
import random
import Putil.np.util as npu
import Putil.tf.util as tfu
import base.logger as plog


root_logger = plog.PutilLogConfig('yolo2ModelBase').logger()
root_logger.setLevel(plog.DEBUG)

Yolo2BuildLogger = root_logger.getChild('Yolo2Build')
Yolo2BuildLogger.setLevel(plog.DEBUG)

Yolo2GenerateFeedLogger = root_logger.getChild('Yolo2GenerateFeed')
Yolo2GenerateFeedLogger.setLevel(plog.DEBUG)

assert tf.__version__ == '1.6.0', Fore.RED + 'version of tensorflow should be 1.6.0'


def append_yolo2_loss(
        yolo2_net_feature,
        class_num,
        prior_h,
        prior_w,
        scalar,
        _dtype=0.32
):
    """
    
    :param yolo2_net_feature: feature from base net output
    :param class_num: the count of the class with background
    :param prior_h: prior height list or 1-D ndarray
    :param prior_w: prior width list or 1-D ndarray
    :param scalar: down sample scalar
    :param _dtype: model parameter dtype, default 0.32
    :return: 
    """
    assert len(prior_w) == len(prior_h), Fore.RED + 'prior height should be same length with prior width'
    print(Fore.YELLOW + '-------generate yolo2 loss---------')
    print(Fore.GREEN + 'class_num : ', class_num)
    print(Fore.GREEN + 'prior_h : ', prior_h)
    print(Fore.GREEN + 'prior_w : ', prior_w)
    print(Fore.GREEN + 'scalar : ', scalar)
    cluster_object_count = len(prior_w)
    place_gt_result = __PlaceGT(cluster_object_count=cluster_object_count, _dtype=_dtype).Place
    place_process_result = __place_process(
        place_gt_result,
        class_num,
        prior_h,
        prior_w,
        scalar=scalar,
        _dtype=_dtype
    )
    pro_result_read_result = __pro_result_reader(
        split_pro_result=yolo2_net_feature,
        cluster_object_count=cluster_object_count)
    calc_iou_result = __calc_iou(
        pro_result_read_result=pro_result_read_result,
        place_process_result=place_process_result,
        scalar=scalar,
        prior_h=prior_h,
        prior_w=prior_w,
        _dtype=_dtype
    )
    loss = __calc_loss(
        split_pro_result=yolo2_net_feature,
        gt_process_result=place_process_result,
        calc_iou_result=calc_iou_result)
    print(Fore.YELLOW + '-------generate yolo2 loss done---------')
    return loss, place_gt_result


# generator placeholder for total feed, designed to easy used and generate
# gt is the standard data
# 'class' : int include background and all kind of object
# 'p_mask' : set 1.0 in the cell location which has an object and set 0.0 for other
# 'n_mask' : set 1.0 in the cell location which does not contain any object and set 0.0 for other
# 'y': object center location y shift from the top left point int the cell, set 0.0 which cell does not contain object
# 'x': object center location x shift from the top left point int the cell, set 0.0 which cell does not contain object
# relationship between real (center_y, center_x, height, width) and (y_shift, x_shift, h_shift, w_shift):
class __PlaceGT:
    def __init__(self, cluster_object_count, _dtype):
        gt_place = dict()
        dtype = tfu.tf_type(_dtype).Type
        with tf.name_scope('GT'):
            gt_place['class'] = tf.placeholder(dtype=tf.int32, shape=[None, None, None, cluster_object_count],
                                               name='class')
            # set 0.0 in the cell which does not contain any object except background
            gt_place['y'] = tf.placeholder(dtype=dtype, shape=[None, None, None, cluster_object_count],
                                           name='y')
            gt_place['x'] = tf.placeholder(dtype=dtype, shape=[None, None, None, cluster_object_count],
                                           name='x')
            # !!!!important: because of the follow process in (__place_process), hw should not contain negative and zero
            # !!!!suggest fill prior value in the cell location which does not contain any object
            gt_place['h'] = tf.placeholder(dtype=dtype, shape=[None, None, None, cluster_object_count],
                                           name='h')
            gt_place['w'] = tf.placeholder(dtype=dtype, shape=[None, None, None, cluster_object_count],
                                           name='w')
            # the mask frequently used in calc loss
            gt_place['p_mask'] = tf.placeholder(dtype=dtype, shape=[None, None, None, 1], name='p_mask')
            gt_place['n_mask'] = tf.placeholder(dtype=dtype, shape=[None, None, None, 1], name='n_mask')
            # avoid learning illegal anchor
            gt_place['anchor_mask'] = tf.placeholder(
                dtype=dtype, shape=[None, None, None, cluster_object_count], name='anchor_mask')
            pass
        self._gt_place = gt_place
        pass

    @property
    def Place(self):
        return self._gt_place

    def __generate(self):
        return self._gt_place
        pass

    @property
    def Class(self):
        return self._gt_place['class']

    @property
    def Y(self):
        return self._gt_place['y']

    @property
    def X(self):
        return self._gt_place['x']

    @property
    def H(self):
        return self._gt_place['h']

    @property
    def W(self):
        return self._gt_place['w']

    @property
    def PMask(self):
        return self._gt_place['p_mask']

    @property
    def NMask(self):
        return self._gt_place['n_mask']

    @property
    def LegalAnchor(self):
        return self._gt_place['anchor_mask']
    pass


# : the pro tensor is not easy to used in calc loss, make same process in this function, this function should make
# : sure gradient can propagate directly
def __split_pro_ac(pro, class_num, cluster_object_count):
    with tf.name_scope('split_and_pro'):
        class_list = list()
        anchor_y_list = list()
        anchor_x_list = list()
        anchor_h_list = list()
        anchor_w_list = list()
        precision_list = list()
        step = 4 + 1 + class_num
        # generate all part y x: sigmoid; h w: None; precision: sigmoid; class: part softmax
        with tf.name_scope('total_split'):
            for i in range(0, cluster_object_count):
                y_part = pro[:, :, :, step * i + 0: step * i + 1]
                anchor_y_list.append(tf.nn.sigmoid(y_part, name='y_{0}_sigmoid'.format(i)))
                x_part = pro[:, :, :, step * i + 1: step * i + 2]
                anchor_x_list.append(tf.nn.sigmoid(x_part, name='x_{0}_sigmoid'.format(i)))
                h_part = pro[:, :, :, step * i + 2: step * i + 3]
                anchor_h_list.append(h_part)
                w_part = pro[:, :, :, step * i + 3: step * i + 4]
                anchor_w_list.append(w_part)
                precision_part = pro[:, :, :, step * i + 4: step * i + 5]
                precision_list.append(tf.nn.sigmoid(precision_part, name='precision_{0}_sigmoid'.format(i)))
                class_part = pro[:, :, :, step * i + 5: step * i + 5 + class_num]
                class_list.append(tf.nn.softmax(class_part, axis=-1, name='class_{0}_softmax'.format(i)))
                pass
            pass
        # generate y x h w pro
        with tf.name_scope('y-x-h-w_pro'):
            y_pro = tf.concat(anchor_y_list, axis=-1, name='y_pro')
            x_pro = tf.concat(anchor_x_list, axis=-1, name='x_pro')
            h_pro = tf.concat(anchor_h_list, axis=-1, name='h_pro')
            w_pro = tf.concat(anchor_w_list, axis=-1, name='w_pro')

        pro_part_list = list()
        anchor_list = list()
        # generate anchor list
        with tf.name_scope('anchor_pro'):
            for i in range(0, cluster_object_count):
                anchor_list.append(
                    tf.concat([anchor_y_list[i], anchor_x_list[i], anchor_h_list[i], anchor_w_list[i]],
                              axis=-1,
                              name='anchor_{0}_concat'.format(i)))
                pass
            anchor_pro = tf.concat(anchor_list, axis=-1, name='anchor_pro')
            pass
        with tf.name_scope('pro'):
            for i in range(0, cluster_object_count):
                pro_part_list.append(
                    tf.concat(
                        [anchor_list[i], precision_list[i], class_list[i]],
                        axis=-1,
                        name='{0}_prediction_gen'.format(i)
                    )
                )
                pass
            pro = tf.concat(pro_part_list, axis=-1, name='pro')
            pass
        with tf.name_scope('class_pro'):
            class_pro = tf.concat(class_list, axis=-1, name='class_pro')
            pass
        with tf.name_scope('precision_pro'):
            precision_pro = tf.concat(precision_list, axis=-1, name='precision_pro')
            pass
        pass
    return {'pro': pro, 'anchor': anchor_pro, 'precision': precision_pro, 'class': class_pro,
            'y': y_pro, 'x': x_pro, 'h': h_pro, 'w': w_pro}
    pass


# : this function is used to generate the standard pro in yolo-version2 network, which split into
# {'pro': pro, 'anchor': anchor_pro, 'precision': precision_pro, 'class': class_pro,
#            'y': y_pro, 'x': x_pro, 'h': h_pro, 'w': w_pro}
def gen_pro(other_new_feature, class_num, cluster_object_count, _dtype=0.32):
    """
    pro = {'pro': pro, 'anchor': anchor_pro, 'precision': precision_pro, 'class': class_pro,
            'y': y_pro, 'x': x_pro, 'h': h_pro, 'w': w_pro}
    :param other_new_feature: base net feature
    :param class_num: 
    :param cluster_object_count: 
    :return: 
    """
    print(Fore.YELLOW + '-----------generate yolo2 base pro---------')
    print(Fore.GREEN + 'class_num : ', class_num)
    print(Fore.GREEN + 'cluster_object_count : ', cluster_object_count)
    feature_chanel = other_new_feature.shape.as_list()[-1]
    dtype = tfu.tf_type(_dtype).Type
    with tf.name_scope('yolo_pro'):
        weight = tf.get_variable(
            name='compress_w',
            shape=[1, 1, feature_chanel, cluster_object_count * (class_num + 4 + 1)],
            initializer=layers.variance_scaling_initializer(seed=0.5, mode='FAN_AVG'),
            dtype=dtype
        )
        bias = tf.get_variable(
            name='compress_b',
            shape=[cluster_object_count * (class_num + 4 + 1)],
            initializer=layers.variance_scaling_initializer(seed=0.5, mode='FAN_AVG'),
            dtype=dtype
        )
        conv = tf.nn.conv2d(other_new_feature, weight, [1, 1, 1, 1], padding='SAME', name='conv')
        add = tf.nn.bias_add(conv, bias, name='bias_add')
        pass
    pro = __split_pro_ac(add, class_num, cluster_object_count)
    return pro
    pass


# : the result of place_gt are not easy to used to calc loss, make some process in the function
def __place_process(gt_place_result, class_num, prior_h, prior_w, scalar, _dtype):
    """
    process the placeholder for using in the network easier
    :param gt_place_result: the result of placeholder
    :param class_num: the count of the class type
    :param prior_h: prior height list
    :param prior_w: prior width list
    :param scalar: down sample scalar
    :return: 
    """
    dtype = tfu.tf_type(_dtype).Type
    gt_process = dict()
    assert len(prior_h) == len(prior_w), Fore.RED + 'len of the prior_h and prior_w should be the same'
    cluster_object_count = len(prior_h)
    with tf.name_scope('gt_place_process'):
        before_one_hot = tf.shape(gt_place_result['class'])
        gt_process_one_hot = tf.one_hot(
            gt_place_result['class'],
            class_num,
            1.0,
            0.0,
            name='one_hot',
            dtype=dtype)
        after_one_hot = tf.shape(gt_process_one_hot)
        reshape_last = tf.multiply(after_one_hot[-1], after_one_hot[-2])
        shape = tf.concat(
            [tf.slice(
                before_one_hot,
                [0],
                [tf.rank(gt_place_result['class']) - 1]
            ),
                [reshape_last]],
            axis=0
        )
        gt_process['class'] = tf.reshape(gt_process_one_hot, shape, name='one_hot_reshape')
        gt_process['feed_class'] = gt_place_result['class']
        gt_process['y'] = gt_place_result['y']
        gt_process['x'] = gt_place_result['x']
        y_pro = tf.div(gt_place_result['y'], scalar)
        x_pro = tf.div(gt_place_result['x'], scalar)
        gt_process['h'] = gt_place_result['h']
        gt_process['w'] = gt_place_result['w']
        h_pro = tf.log(tf.div(gt_place_result['h'], prior_h))
        w_pro = tf.log(tf.div(gt_place_result['w'], prior_w))
        # generate entirety anchor
        shape = tf.concat([tf.slice(tf.shape(gt_process['h']), [0], [3]), [4 * cluster_object_count]], axis=0)
        gt_process['anchor'] = tf.reshape(
            tf.concat([tf.reshape(y_pro, [-1, cluster_object_count, 1]),
                       tf.reshape(x_pro, [-1, cluster_object_count, 1]),
                       tf.reshape(h_pro, [-1, cluster_object_count, 1]),
                       tf.reshape(w_pro, [-1, cluster_object_count, 1])], axis=0),
            shape)
        gt_process['anchor_obj_mask'] = tf.multiply(gt_place_result['p_mask'], gt_place_result['anchor_mask'])
        gt_process['p_mask'] = gt_place_result['p_mask']
        gt_process['n_mask'] = gt_place_result['n_mask']
        gt_process['anchor_mask'] = gt_place_result['anchor_mask']
        pass
    return gt_process


# : to read the pro result, avoid the gradient propagate from precision loss to the network twice
def __pro_result_reader(split_pro_result, cluster_object_count):
    """
    read the pro result, avoid the gradient propagate from precision loss to the network twice
    :param split_pro_result: __split_pro result
    :param cluster_object_count: prior cluster count
    :return: 
    """
    pro_result_read = dict()
    pro_result_read['y'] = tf.identity(split_pro_result['y'], name='y_read')
    pro_result_read['x'] = tf.identity(split_pro_result['x'], name='x_read')
    pro_result_read['h'] = tf.identity(split_pro_result['h'], name='h_read')
    pro_result_read['w'] = tf.identity(split_pro_result['w'], name='w_read')
    return pro_result_read
    pass


# :use gt_anchor and anchor_pro to calc iou， output for calc precision loss
def __calc_iou(pro_result_read_result, place_process_result, scalar, prior_h, prior_w, _dtype):
    yt = place_process_result['y']
    xt = place_process_result['x']
    ht = place_process_result['h']
    wt = place_process_result['w']
    anchor_mask = place_process_result['anchor_mask']
    p_mask = place_process_result['p_mask']
    n_mask = place_process_result['n_mask']
    dtype = tfu.tf_type(_dtype).Type
    with tf.name_scope('calc_iou'):
        with tf.name_scope('p_iou'):
            yp = pro_result_read_result['y'] * scalar
            xp = pro_result_read_result['x'] * scalar
            hp = tf.multiply(tf.exp(pro_result_read_result['h']), prior_h)
            wp = tf.multiply(tf.exp(pro_result_read_result['w']), prior_w)
            p_iou = tf.multiply(
                tf.nn.relu(tf.subtract(tf.multiply(2.0, tf.abs(tf.subtract(hp, ht))), tf.abs(tf.subtract(yp, yt)))),
                tf.nn.relu(tf.subtract(tf.multiply(2.0, tf.abs(tf.subtract(wp, wt))), tf.abs(tf.subtract(xp, xt)))),
                name='p_iou'
            )
            pass
        with tf.name_scope('n_iou'):
            shape = p_iou.get_shape().as_list()
            n_iou = tf.zeros(shape=shape, dtype=dtype, name='n_iou')
            pass
        iou = tf.add(
            tf.multiply(
                p_iou,
                anchor_mask,
                name='apply_p_mask'
            ),
            tf.multiply(
                n_iou,
                -1 * (1 - anchor_mask),
                name='apply_n_mask'
            ),
            'iou_label'
        )
        pass
    return iou
    pass


# : generate the loss op
def __calc_loss(split_pro_result, gt_process_result, calc_iou_result):
    anchor_loss_weight = 1.0
    precision_loss_weight = 1.0
    class_loss_weight = 1,0
    lambda_obj = 1.0
    lambda_noobj = 0.1
    y_pro = split_pro_result['y']
    x_pro = split_pro_result['x']
    h_pro = split_pro_result['h']
    w_pro = split_pro_result['w']
    precision_pro = split_pro_result['precision']
    class_pro = split_pro_result['class']
    anchor_mask = gt_process_result['anchor_mask']
    gt_y = gt_process_result['y']
    gt_x = gt_process_result['x']
    gt_h = gt_process_result['h']
    gt_w = gt_process_result['w']
    gt_class = gt_process_result['class']

    legal_anchor_amount = tf.reduce_sum(anchor_mask, name='legal_anchor_amount')

    with tf.name_scope('loss'):
        with tf.name_scope('anchor_loss'):
            # yx loss part
            with tf.name_scope('yx_loss'):
                yx_loss = tf.add(
                    tf.square(tf.multiply(tf.subtract(y_pro, gt_y, name='y_sub'), anchor_mask, name='apply_p_mask'), name='y_square'),
                    tf.square(tf.multiply(tf.subtract(x_pro, gt_x, name='x_sub'), anchor_mask, name='apply_p_mask'),name='x_square'),
                    name='y_x_add')
                pass
            # hw loss part
            with tf.name_scope('hw_loss'):
                hw_loss = tf.add(
                    tf.square(
                        tf.subtract(
                            tf.sqrt(tf.multiply(h_pro, anchor_mask, name='h_pro_apply_p_mask'), name='h_pro_sqrt'),
                            tf.sqrt(tf.multiply(gt_h, anchor_mask, name='gt_h_apply_p_mask'), name='gt_h_sqrt'),
                            name='h_sub'),
                        name='h_square'),
                    tf.square(
                        tf.subtract(
                            tf.sqrt(tf.multiply(w_pro, anchor_mask, name='w_pro_apply_p_mask'), name='w_pro_sqrt'),
                            tf.sqrt(tf.multiply(gt_w, anchor_mask, name='gt_w_apply_p_mask'), name='gt_w_sqrt'),
                            name='w_sub'),
                        name='w_square'),
                    name='hw_add')
                pass

            # anchor loss
            anchor_loss = tf.add(
                tf.multiply(
                    lambda_obj,
                    tf.div(
                        tf.reduce_sum(tf.multiply(yx_loss, gt_process_result['anchor_obj_mask']), name='batch_sum'),
                        legal_anchor_amount,
                        name='yx_anchor_obj_mean'),
                    name='apply_lambda_weight'),
                tf.multiply(
                    lambda_obj,
                    tf.div(
                        tf.reduce_sum(tf.multiply(hw_loss, gt_process_result['anchor_obj_mask']), name='batch_sum'),
                        legal_anchor_amount,
                        name='hw_anchor_obj_mean'),
                    name='apply_lambda_weight'),
                name='anchor_loss_sum'
            )
            anchor_loss = tf.multiply(anchor_loss, anchor_loss_weight, name='apply_anchor_loss_weight')
            pass
        with tf.name_scope('precision_loss'):
            precision_loss = tf.reduce_mean(tf.square(tf.subtract(precision_pro, calc_iou_result)))
            precision_loss = tf.multiply(precision_loss, precision_loss_weight, name='apply_precision_loss_weight')
            # precision_loss = tf.add(p_precision_loss, n_precision_loss, name='loss')
            pass
        with tf.name_scope('class_loss'):
            anchor_amount = anchor_mask.get_shape().as_list()[3]
            shape = tf.concat([[-1], [tf.div(tf.shape(gt_class)[-1], anchor_amount)]], axis=0)
            # class_pro_reshape = class_pro.get_shape().as_list()[0:3] + [anchor_amount, class_amount]
            class_loss_whole = tf.multiply(
                tf.reshape(tf.square(tf.subtract(gt_class, class_pro)), shape),
                tf.reshape(anchor_mask, [-1, 1]),
                name='class_loss')
            class_loss = tf.multiply(
                lambda_obj,
                tf.div(tf.reduce_sum(class_loss_whole, name='batch_sum'), legal_anchor_amount, name='class_anchor_obj_mean'),
                name='apply_lambda_weight')
            class_loss = tf.multiply(class_loss, class_loss_weight, name='apply_class_loss_weight')
            pass
        total_loss = tf.add(anchor_loss, tf.add(precision_loss, class_loss), name='total_loss')
        pass
    return total_loss
    pass


import six
import abc


@six.add_metaclass(abc.ABCMeta)
class Yolo2GenerateI(object):
    """
    use normal information to generate the tensor feeding into the network build with above function
    generate: y, x, w, h, class, obj_mask, nobj_mask, anchor_mask
    """
    @abc.abstractmethod
    def _default_generate_feed_function(self, param):
        pass

    @abc.abstractmethod
    def CheckGenerateFeedParamFit(self, param):
        pass

    @abc.abstractmethod
    def _default_generate_result_function(self, param):
        pass

    @abc.abstractmethod
    def CheckGenerateResultParamFit(self, param):
        pass

    pass


@six.add_metaclass(abc.ABCMeta)
class Yolo2Generate(Yolo2GenerateI):
    def __init__(self):
        self._generate_feed_function = None
        self._generate_result_function = None
        pass

    def GenerateFeed(self, param):
        return self._generate_feed_function(param)
        pass

    def GenerateResult(self, param):
        return self._generate_result_function(param)
        pass

    def InstallGenerateFeedFunction(self, analysis_function):
        self._generate_feed_function = analysis_function
        pass

    def InstallGenerateResultFunction(self, analysis_function):
        self._generate_result_function = analysis_function
        pass
    pass


"""
StandardYolo2Generate: 
    the paper use:
        the center of prior anchor is locate at (i * scalar, j * scalar)
        
        anchor mask:[batch, cell_height, cell_width, prior_anchor_amount]
            every obj get one and only one nearest anchor to predict
            any anchor does not hold an obj would be rejected, place zero
            any anchor cross the edge of image was reject， place zero
*************************************important*************************************            
            (some conditions:
                if more than one prior anchor Iou gt_x, get the same maximum value,
                they would be hold to the gt_x at the same time
                if no more prior to provide gt_prediction this gt would be abandon)
#####################################important#####################################

        obj mask:[batch, cell_height, cell_width, prior_anchor_amount] 
            any cell does not hold any obj place zero
            any cell does hold any obj place one
            
        nobj mask:[batch, cell_height, cell_width, prior_anchor_amount]
            any cell does not hold any obj place one
            any cell does hold any obj place zero
            
        y:[batch, cell_height, cell_width, prior_anchor_amount]
            y = (real_center_y % scalar) / scalar
            
        x:[batch, cell_height, cell_width, prior_anchor_amount]
            x = (real_center_x % scalar) / scalar
            
        h:[batch, cell_height, cell_width, prior_anchor_amount]
            h = ln(real_height / prior_height)
            
        w:[batch, cell_height, cell_width, prior_anchor_amount]
            w = ln(real_width / prior_width)
            
        class:[batch, cell_height, cell_width, prior_anchor_amount]
            class = obj_represent_int
"""
import Putil.calc.estimate as es


class StandardYolo2Generate(Yolo2Generate):
    def __init__(self, prior_hw, scalar, _dtype):
        Yolo2Generate.__init__(self)

        self.feed_height = None
        self.feed_width = None
        self.batch = None

        self.y = None
        self.x = None
        self.h = None
        self.w = None
        self.cross_anchor_reject = None
        self.anchor_mask = None
        self.classify = None
        self.obj_mask = None
        self.nobj_mask = None

        self.scalar = scalar
        self._dtype = _dtype
        self.prior_hw = prior_hw
        self.anchor_mask = len(prior_hw)
        pass

    def __update_feed_shape(self, batch, feed_height, feed_width, anchor_amount, _dtype):
        """
        while any parameter include batch, feed_height, feed_width, anchor_amount, _dtype
        changed, the tensor shape or type should be changed
        this is the function for updating the tensor
        :return:
        """
        self._dtype = _dtype
        self.batch = batch
        self.feed_height = feed_height
        self.feed_width = feed_width
        self.anchor_amount = anchor_amount
        dtype = npu.np_type(self._dtype).Type
        self.y = np.zeros(shape=[self.batch, self.feed_height, self.feed_width, self.anchor_amount], dtype=dtype)
        self.x = np.zeros(shape=[self.batch, self.feed_height, self.feed_width, self.anchor_amount], dtype=dtype)
        self.h = np.zeros(shape=[self.batch, self.feed_height, self.feed_width, self.anchor_amount], dtype=dtype)
        self.w = np.zeros(shape=[self.batch, self.feed_height, self.feed_width, self.anchor_amount], dtype=dtype)
        self.anchor_mask = np.ones(shape=[self.batch, self.feed_height, self.feed_width, self.anchor_amount], dtype=dtype)
        self.obj_mask = np.zeros(shape=[self.batch, self.feed_height, self.feed_width, 1], dtype=dtype)
        self.nobj_mask = np.zeros(shape=[self.batch, self.feed_height, self.feed_width, 1], dtype=dtype)
        self.classify = np.zeros(shape=[self.batch, self.feed_height, self.feed_width, self.anchor_amount], dtype=dtype)
        self.cross_anchor_reject = self.__cross_edge_anchor_reject()
        pass

    def __cross_edge_anchor_reject(self):
        """
        process the anchor mask to make the anchor which cross the edge of image zero
        :param batch:
        :param feed_height:
        :param feed_width:
        :param anchor_amount:
        :param _dtype:
        :return:
        """
        anchor_mask_shape = self.anchor_mask.shape
        # calculate the allowed band for box to expand in the anchor mask[batch, feed_height, feed_width, anchor_amount, 4]
        # 4: [top_height_band, bottom_height_band, left_width_band, right_width_band]
        max_band_h = anchor_mask_shape[1]
        max_band_w = anchor_mask_shape[2]
        top_height_band = np.expand_dims(np.linspace(0, max_band_h - 1, num=max_band_h).repeat(max_band_w).reshape([max_band_h, max_band_w]), -1)
        bottom_height_band = np.expand_dims(np.linspace(0, max_band_h - 1, num=max_band_h)[::-1].repeat(max_band_w).reshape([max_band_h, max_band_w]), -1)
        left_width_band = np.expand_dims(np.linspace(0, max_band_w - 1, num=max_band_w).repeat(max_band_h).reshape([max_band_w, max_band_h]).T, -1)
        right_width_band = np.expand_dims(np.linspace(0, max_band_w - 1, num=max_band_w)[::-1].repeat(max_band_h).reshape([max_band_w, max_band_h]).T, -1)
        band_t_b_l_r = np.concatenate((top_height_band, bottom_height_band, left_width_band, right_width_band), -1)
        # calculate the prior wh expand in the anchor mask[batch, feed_height, feed_width, anchor_amount, 4]
        # subtract, replace negative by zero, multiply, and then concat
        prior_expand_t_b_1_r_list = []
        rejected = []
        for i in self.prior_hw:
            expand = np.array([0.5 * i[0], 0.5 * i[0], 0.5 * i[1], 0.5 * i[1]]).repeat(max_band_h * max_band_w).reshape(max_band_h, max_band_w, 4)
            expanded = band_t_b_l_r - expand
            expanded[expanded < 0] = 0
            rejected.append(expanded[:, :, 0] * expanded[:, :, 1] * expanded[:, :, 2] * expanded[:, :, 3])
            pass
        rejected = np.concatenate(rejected)
        # multiply the anchor mask
        return rejected
        pass

    def __find_same_cell_location(self, scalar, gt_box):
        """
        use scalar and gt_box to generate same cell format
        [[[gt_box, ...](the box in the same cell, [cell]], [[offset]]...]
        gt_box: [y, x, h, w]; cell: [cell_y=gt_box.y//scalar, cell_x=gt_box.x//scalar];
        offset: [offset_y=gt_box.y%scalar, offset_x=gt_box.x%scalar]
        :param scalar:
        :param gt_box:
        :return:
        """
        format = list()
        # sort by y**2 + x**2 get the index
        array_gt_box = np.array(gt_box)
        order = (array_gt_box[:, 0] ** 2 + array_gt_box[:, 1] ** 2).argsort()
        killed = []
        for i in range(0, len(order)):
            if i in killed:
                continue
            cell_y = gt_box[i][0] // scalar
            cell_x = gt_box[i][1] // scalar
            offset_y = gt_box[i][0] % scalar
            offset_x = gt_box[i][1] % scalar
            format.append([[]])
            format[-1][0].append(gt_box[i])
            format[-1].append([cell_y, cell_x])
            format[-1].append([])
            format[-1][-1].append([offset_y, offset_x])
            for j in range(i + 1, len(order)):
                if (gt_box[i][0] // scalar == gt_box[j][0] // scalar) & (
                        gt_box[i][1] // scalar == gt_box[j][1] // scalar):
                    # add to the format and add to killed
                    offset_y = gt_box[j][0] % scalar
                    offset_x = gt_box[j][1] % scalar
                    format[-1][0].append(gt_box[j])
                    format[-1][-1].append([offset_y, offset_x])
                    killed.append(j)
                    pass
                else:
                    break
                    pass
                pass
            pass
        return format
        pass

    def _default_generate_feed_function(self, param):
        """

        :param param: dict
            gt_box: support batch [[obj_0_yxhwc, obj_1_yxhwc, ...obj_n_yxhwc..., ], sample_1, sample_2, ....]
            prior_hw: [[h0, w0], prior_1_h2, ...prior_n_hw...]
            dtype: data type , float , base on np.np_type
            scalar: net work pool scalar , static parameter: int
            anchor_reject_base_iou: a value while the prior box IoU gt box lower than it and not the max iou prior box,
                the prior would be reject to be added to training, float
            image_height: batch accordant: int
            image_width: batch accordant: int
            shape_policy: batch accordant: string {'down_clip', 'up_clip', 'down_fit', 'up_fit'}
        :return:
        """
        ret = dict()
        gt_box = param['gt_box']
        iou_reject = param['iou_reject']
        feed_height = param['feed_height']
        feed_width = param['feed_width']
        batch = len(gt_box)

        if (batch != self.batch) \
                or (feed_height != self.feed_height or feed_width != self.feed_width) \
                or (self.feed_height is None and self.feed_width is None):
            self.__update_feed_shape(batch, feed_height, feed_width, self.anchor_amount, self._dtype)

        gt_format = self.__find_same_cell_location(scalar=self.scalar, gt_box=gt_box)

        for i in gt_format:
            cell_obj_amount = len(i[0])
            ohw = np.concatenate([i[0], i[2]], -1)
            iou_matrix = es.calc_iou_matrix_ohw(
                self.prior_hw,
                ohw,
                group1_h_index=2,
                group1_w_index=3,
                group2_y_index=4,
                group2_x_index=5,
                group2_h_index=2,
                group2_w_index=3
            )
            anchor_activate = []
            for j in range(0, cell_obj_amount):
                max_iou = np.max(iou_matrix)
                if max_iou == 0.0:
                    break
                location = np.where(iou_matrix == np.max(iou_matrix))
                for k in zip(location[0], location[1]):
                    iou_matrix
                    pass
                pass
            pass
        pass

    @property
    def FindSameCellLocation(self):
        return self.__find_same_cell_location
        pass

    def __divide_anchor(self, gt_format_item):
        """
        use the output of __find_same_cell_location to divide anchor's owner
        :param gt_format:
        :return:
        """

        pass

    def CheckGenerateFeedParamFit(self, param):
        pass

    def _default_generate_result_function(self, param):
        pass

    def CheckGenerateResultParamFit(self, param):
        pass
    pass
