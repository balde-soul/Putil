# coding = utf-8
import tensorflow as tf
import tensorflow.contrib.layers as layers
from colorama import Fore
import numpy as np
import random
assert tf.__version__ == '1.6.0', Fore.RED + 'version of tensorflow should be 1.6.0'


def append_yolo2_loss(
        yolo2_net_feature,
        class_num,
        prior_h,
        prior_w,
        scalar
):
    """
    
    :param yolo2_net_feature: feature from base net output
    :param class_num: the count of the class with background
    :param prior_h: prior height list or 1-D ndarray
    :param prior_w: prior width list or 1-D ndarray
    :param scalar: down sample scalar
    :return: 
    """
    assert len(prior_w) == len(prior_h), Fore.RED + 'prior height should be same length with prior width'
    print(Fore.YELLOW + '-------generate yolo2 loss---------')
    print(Fore.GREEN + 'class_num : ', class_num)
    print(Fore.GREEN + 'prior_h : ', prior_h)
    print(Fore.GREEN + 'prior_w : ', prior_w)
    print(Fore.GREEN + 'scalar : ', scalar)
    cluster_object_count = len(prior_w)
    place_gt_result = __PlaceGT(cluster_object_count=cluster_object_count).Place
    place_process_result = __place_process(
        place_gt_result,
        class_num,
        prior_h,
        prior_w,
        scalar=scalar)
    pro_result_read_result = __pro_result_reader(
        split_pro_result=yolo2_net_feature,
        cluster_object_count=cluster_object_count)
    calc_iou_result = __calc_iou(
        pro_result_read_result=pro_result_read_result,
        place_process_result=place_process_result,
        scalar=scalar,
        prior_h=prior_h,
        prior_w=prior_w)
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
    def __init__(self, cluster_object_count):
        gt_place = dict()
        with tf.name_scope('GT'):
            gt_place['class'] = tf.placeholder(dtype=tf.int32, shape=[None, None, None, cluster_object_count],
                                               name='class')
            # set 0.0 in the cell which does not contain any object except background
            gt_place['y'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count],
                                           name='y')
            gt_place['x'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count],
                                           name='x')
            # !!!!important: because of the follow process in (__place_process), hw should not contain negative and zero
            # !!!!suggest fill prior value in the cell location which does not contain any object
            gt_place['h'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count],
                                           name='h')
            gt_place['w'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count],
                                           name='w')
            # the mask frequently used in calc loss
            gt_place['p_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='p_mask')
            gt_place['n_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1], name='n_mask')
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
def gen_pro(other_new_feature, class_num, cluster_object_count):
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
    with tf.name_scope('yolo_pro'):
        weight = tf.get_variable(
            name='compress_w', shape=[1, 1, feature_chanel, cluster_object_count * (class_num + 4 + 1)],
            initializer=layers.variance_scaling_initializer(seed=0.5, mode='FAN_AVG'), dtype=tf.float32)
        bias = tf.get_variable(
            name='compress_b', shape=[cluster_object_count * (class_num + 4 + 1)],
            initializer=layers.variance_scaling_initializer(seed=0.5, mode='FAN_AVG')
        )
        conv = tf.nn.conv2d(other_new_feature, weight, [1, 1, 1, 1], padding='SAME', name='conv')
        add = tf.nn.bias_add(conv, bias, name='bias_add')
        pass
    pro = __split_pro_ac(add, class_num, cluster_object_count)
    return pro
    pass


# : the result of place_gt are not easy to used to calc loss, make some process in the function
def __place_process(gt_place_result, class_num, prior_h, prior_w, scalar):
    """
    process the placeholder for using in the network easier
    :param gt_place_result: the result of placeholder
    :param class_num: the count of the class type
    :param prior_h: prior height list
    :param prior_w: prior width list
    :param scalar: down sample scalar
    :return: 
    """
    gt_process = dict()
    assert len(prior_h) == len(prior_w), Fore.RED + 'len of the prior_h and prior_w should be the same'
    cluster_object_count = len(prior_h)
    with tf.name_scope('gt_place_process'):
        before_one_hot = tf.shape(gt_place_result['class'])
        gt_process_one_hot = tf.one_hot(gt_place_result['class'], class_num, 1.0, 0.0, name='one_hot')
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
        gt_process['p_mask'] = gt_place_result['p_mask']
        gt_process['n_mask'] = gt_place_result['n_mask']
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
def __calc_iou(pro_result_read_result, place_process_result, scalar, prior_h, prior_w):
    yt = place_process_result['y']
    xt = place_process_result['x']
    ht = place_process_result['h']
    wt = place_process_result['w']
    with tf.name_scope('calc_iou'):
        yp = pro_result_read_result['y'] * scalar
        xp = pro_result_read_result['x'] * scalar
        hp = tf.multiply(tf.exp(pro_result_read_result['h']), prior_h)
        wp = tf.multiply(tf.exp(pro_result_read_result['w']), prior_w)
        cross_h = tf.multiply(
            tf.nn.relu(tf.subtract(tf.multiply(2.0, tf.abs(tf.subtract(hp, ht))), tf.abs(tf.subtract(yp, yt)))),
            tf.nn.relu(tf.subtract(tf.multiply(2.0, tf.abs(tf.subtract(wp, wt))), tf.abs(tf.subtract(xp, xt)))),
            name='t_iou'
        )
    return cross_h
    pass


# : generate the loss op
def __calc_loss(split_pro_result, gt_process_result, calc_iou_result):
    lambda_obj = 1.0
    lambda_noobj = 0.1
    y_pro = split_pro_result['y']
    x_pro = split_pro_result['x']
    h_pro = split_pro_result['h']
    w_pro = split_pro_result['w']
    anchor_pro = split_pro_result['anchor']
    precision_pro = split_pro_result['precision']
    class_pro = split_pro_result['class']
    p_mask = gt_process_result['p_mask']
    n_mask = gt_process_result['n_mask']
    gt_y = gt_process_result['y']
    gt_x = gt_process_result['x']
    gt_h = gt_process_result['h']
    gt_w = gt_process_result['w']
    gt_anchor = gt_process_result['anchor']
    # gt_precision = place_gt_result['precision']
    gt_class = gt_process_result['class']
    with tf.name_scope('loss'):
        with tf.name_scope('anchor_loss'):
            # yx loss part
            with tf.name_scope('yx_loss'):
                yx_loss = tf.add(
                    tf.square(
                        tf.multiply(
                            tf.subtract(
                                y_pro,
                                gt_y,
                                name='y_sub'
                            ),
                            p_mask,
                            name='apply_p_mask'
                        ),
                        name='y_square'
                    ),
                    tf.square(
                        tf.multiply(
                            tf.subtract(
                                x_pro,
                                gt_x,
                                name='x_sub'
                            ),
                            p_mask,
                            name='apply_p_mask'
                        ),
                        name='x_square'
                    ),
                    name='y_x_add'
                )
                # yx_loss = tf.multiply(
                #     tf.add(
                #         tf.square(
                #             tf.subtract(
                #                 y_pro,
                #                 gt_y,
                #                 name='y_sub'
                #             ),
                #             name='y_square'
                #         ),
                #         tf.square(
                #             tf.subtract(
                #                 x_pro,
                #                 gt_x,
                #                 name='x_sub'
                #             ),
                #             name='x_square'
                #         ),
                #         name='y_x_add'
                #     ),
                #     p_mask,
                #     name='apply_p_mask'
                # )
                pass
            # hw loss part
            with tf.name_scope('hw_loss'):
                hw_loss = tf.add(
                    tf.square(
                        tf.subtract(
                            tf.sqrt(
                                tf.multiply(
                                    h_pro,
                                    p_mask,
                                    name='h_pro_apply_p_mask'
                                ),
                                name='h_pro_sqrt'
                            ),
                            tf.sqrt(
                                tf.multiply(
                                    gt_h,
                                    p_mask,
                                    name='gt_h_apply_p_mask'
                                ),
                                name='gt_h_sqrt'
                            ),
                            name='h_sub'
                        ),
                        name='h_square'
                    ),
                    tf.square(
                        tf.subtract(
                            tf.sqrt(
                                tf.multiply(
                                    w_pro,
                                    p_mask,
                                    name='w_pro_apply_p_mask'
                                ),
                                name='w_pro_sqrt'
                            ),
                            tf.sqrt(
                                tf.multiply(
                                    gt_w,
                                    p_mask,
                                    name='gt_w_apply_p_mask'
                                ),
                                name='gt_w_sqrt'
                            ),
                            name='w_sub'
                        ),
                        name='w_square'
                    ),
                    name='hw_add'
                )
                pass
            # p_anchor = tf.multiply(
            #     tf.square(
            #         tf.subtract(
            #             gt_anchor,
            #             anchor_pro
            #         )
            #     ),
            #     p_mask,
            #     name='p_anchor'
            # )
            # p_anchor_loss = tf.multiply(
            #     lambda_obj,
            #     tf.reduce_sum(tf.reduce_mean(p_anchor, axis=0)),
            #     name='p_loss'
            # )
            # n_anchor = tf.multiply(
            #     tf.square(
            #         tf.subtract(
            #             gt_anchor,
            #             anchor_pro
            #         )
            #     ),
            #     n_mask,
            #     name='n_anchor'
            # )
            # n_anchor_loss = tf.multiply(
            #     lambda_noobj,
            #     tf.reduce_sum(tf.reduce_mean(n_anchor, axis=0)),
            #     name='n_loss'
            # )
            # anchor_loss = tf.add(p_anchor_loss, n_anchor_loss, name='loss')

            # anchor loss
            anchor_loss = tf.add(
                tf.multiply(
                    lambda_obj,
                    tf.reduce_sum(
                        yx_loss,
                        name='yx_sum'
                    ),
                    name='apply_lambda_weight'
                ),
                tf.multiply(
                    lambda_obj,
                    tf.reduce_sum(
                        hw_loss,
                        name='hw_sum'
                    ),
                    name='apply_lambda_weight'
                ),
                name='anchor_sum'
            )
            pass
        with tf.name_scope('precision_loss'):
            # n_precision = tf.multiply(
            #     tf.square(
            #         tf.subtract(
            #             precision_pro,
            #             0
            #         )
            #     ),
            #     n_mask,
            #     name='n_precision'
            # )
            # n_precision_loss = tf.multiply(
            #     tf.reduce_sum(tf.reduce_mean(n_precision, axis=0)),
            #     lambda_noobj,
            #     name='n_loss'
            # )
            p_precision = tf.multiply(
                tf.square(
                    tf.subtract(
                        precision_pro,
                        calc_iou_result
                    )
                ),
                p_mask,
                name='p_precision'
            )
            p_precision_loss = tf.multiply(
                tf.reduce_sum(tf.reduce_mean(p_precision, axis=0)),
                lambda_obj,
                name='p_loss'
            )
            precision_loss = p_precision_loss
            # precision_loss = tf.add(p_precision_loss, n_precision_loss, name='loss')
            pass
        with tf.name_scope('class_loss'):
            p_class = tf.multiply(
                tf.square(
                    tf.subtract(
                        gt_class,
                        class_pro
                    )
                ),
                p_mask,
                name='p_class'
            )
            p_class_loss = tf.multiply(
                lambda_obj,
                tf.reduce_sum(tf.reduce_mean(p_class, axis=0)),
                name='p_loss'
            )
            n_class = tf.multiply(
                tf.square(
                    tf.subtract(
                        gt_class,
                        class_pro
                    )
                ),
                n_mask,
                name='n_class'
            )
            n_class_loss = tf.multiply(
                lambda_noobj,
                tf.reduce_sum(tf.reduce_mean(n_class, axis=0)),
                name='n_loss'
            )
            class_loss = tf.add(p_class_loss, n_class_loss, name='loss')
            pass
        total_loss = tf.add(anchor_loss, tf.add(precision_loss, class_loss), name='total_loss')
        pass
    return total_loss
    pass


if __name__ == '__main__':
    feature_feed = tf.placeholder(dtype=tf.float32, shape=[10, 10, 10, 100], name='other_net_feature')
    f = np.zeros([10, 10, 10, 100], np.float32)
    p_mask = np.zeros([10, 10, 10, 1], np.float32)
    n_mask = np.ones([10, 10, 10, 1], np.float32)
    cl = np.zeros([10, 10, 10, 4], np.int64)
    y = np.zeros([10, 10, 10, 4], np.float32)
    x = np.zeros([10, 10, 10, 4], np.float32)
    h = np.zeros([10, 10, 10, 4], np.float32)
    w = np.zeros([10, 10, 10, 4], np.float32)
    yolo_feature = gen_pro(feature_feed, 3, 4)
    loss, place = append_yolo2_loss(yolo_feature, 3, [10, 5, 3, 4], [2, 3, 4, 8], 32)
    tf.summary.FileWriter('../test/yolo/model_base-', tf.Session().graph).close()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print(sess.run([loss], feed_dict={
        feature_feed: f,
        place['class']: cl,
        place['y']: y,
        place['x']: x,
        place['h']: h,
        place['w']: w,
        place['p_mask']: p_mask,
        place['n_mask']: n_mask
    }))
    pass
