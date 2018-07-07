# coding = utf-8
import tensorflow as tf
import tensorflow.contrib.layers as layers
from colorama import Fore
assert tf.__version__ == '1.6.0', Fore.RED + 'version of tensorflow should be 1.6.0'


def append_yolo2(
        other_new_feature,
        feature_chanel,
        class_num,
        prior_h,
        prior_w,
        scalar,
):
    """
    
    :param other_new_feature: feature from base net output
    :param feature_chanel: other_new_feature channel
    :param class_num: the count of the class with background
    :param prior_h: prior height list or 1-D ndarray
    :param prior_w: prior width list or 1-D ndarray
    :param scalar: down sample scalar
    :return: 
    """
    assert len(prior_w) == len(prior_h), Fore.RED + 'prior height should be same length with prior width'
    cluster_object_count = len(prior_w)
    pro = gen_pro(other_new_feature, feature_chanel, class_num, cluster_object_count)
    split_pro_result = split_pro(pro, class_num=class_num, cluster_object_count=cluster_object_count)
    place_gt_result = PlaceGT(cluster_object_count=cluster_object_count).Place
    place_process_result = place_process(place_gt_result, class_num, prior_h, prior_w, scalar=scalar)
    pro_result_read_result = pro_result_reader(split_pro_result=split_pro_result,
                                               cluster_object_count=cluster_object_count)
    calc_iou_result = calc_iou(pro_result_read_result=pro_result_read_result, place_process_result=place_process_result,
                               scalar=scalar, prior_h=prior_h, prior_w=prior_w)
    loss = calc_loss(split_pro_result=split_pro_result, gt_process_result=place_process_result,
                     calc_iou_result=calc_iou_result)
    return loss, place_gt_result



# todo: generator placeholder for total feed, designed to easy used and generate
# gt is the standard data
# 'class' : one_hot include background and all kind of object
# 'p_mask' : set 1.0 in the cell location which has an object and set 0.0 for other
# 'n_mask' : set 1.0 in the cell location which does not contain any object and set 0.0 for other
# relationship between real (center_y, center_x, height, width) and (y_shift, x_shift, h_shift, w_shift):
class PlaceGT:
    def __init__(self, cluster_object_count):
        gt_place = dict()
        with tf.name_scope('GT'):
            gt_place['class'] = tf.placeholder(dtype=tf.int32, shape=[None, None, None, cluster_object_count])
            # set 0.0 in the cell which does not contain any object except background
            gt_place['y'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count])
            gt_place['x'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count])
            # !!!!important: because of the follow process in (place_process), hw should not contain negative and zero
            # !!!!suggest fill prior value in the cell location which does not contain any object
            gt_place['h'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count])
            gt_place['w'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, cluster_object_count])
            # the mask frequently used in calc loss
            gt_place['p_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
            gt_place['n_mask'] = tf.placeholder(dtype=tf.float32, shape=[None, None, None, 1])
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


# todo: this function is used to generate the standard pro in yolo-version2 network
def gen_pro(other_new_feature, feature_chanel, class_num, cluster_object_count):
    with tf.name_scope('yolo_pro'):
        weight = tf.get_variable(
            name='compress_w', shape=[1, 1, feature_chanel, cluster_object_count * (class_num + 4 + 1)],
            initializer=layers.variance_scaling_initializer(seed=0.5, mode='FAN_AVG'), dtype=tf.float32)
        bias = tf.get_variable(
            name='compress_b', shape=[cluster_object_count],
            initializer=layers.variance_scaling_initializer(seed=0.5, mode='FAN_AVG')
        )
        conv = tf.nn.conv2d(other_new_feature, weight, [1, 1, 1, 1], padding='SAME', name='pro')
    return conv
    pass


# todo: the pro tensor is not easy to used in calc loss, make same process in this function, this function should make
# todo: sure gradient can propagate directly
def split_pro(pro, class_num, cluster_object_count):
    with tf.name_scope('split'):
        class_list = list()
        anchor_list = list()
        precision_list = list()
        step = 4 + 1 + class_num
        for i in range(0, cluster_object_count):
            class_list.append(tf.slice(pro, [0, 0, 0, step * i + 5], [-1, -1, -1, class_num]))
            anchor_list.append(tf.slice(pro, [0, 0, 0, step * i], [-1, -1, -1, 4]))
            precision_list.append(tf.slice(pro, [0, 0, 0, step * i + 4], [-1, -1, -1, 1]))
            pass
        class_pro = tf.concat(class_list, axis=-1, name='class_pro')
        anchor_pro = tf.concat(anchor_list, axis=-1, name='anchor_pro')
        precision_pro = tf.concat(precision_list, axis=-1, name='precision_pro')
        pass
    return {'anchor': anchor_pro, 'precision': precision_pro, 'class': class_pro}
    pass


# todo: the result of place_gt are not easy to used to calc loss, make some process in the function
def place_process(gt_place_result, class_num, prior_h, prior_w, scalar):
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
        gt_process_one_hot = tf.one_hot(gt_place_result['class'], class_num, 1.0, 0.0, axis=-1)
        after_one_hot = tf.shape(gt_process_one_hot)
        reshape_last = tf.multiply(after_one_hot[-1], after_one_hot[-2])
        shape = tf.concat(
            [tf.slice(
                tf.transpose(before_one_hot),
                [1],
                [-1]
            ),
                [gt_process_one_hot.get_shape().as_list()[-1] * gt_process_one_hot.get_shape().as_list()[-2] ]],
            axis=0
        )
        gt_process['class'] = tf.reshape(gt_process_one_hot, shape)
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


# todo: to read the pro result, avoid the gradient propagate from precision loss to the network twice
def pro_result_reader(split_pro_result, cluster_object_count):
    """
    read the pro result, avoid the gradient propagate from precision loss to the network twice
    :param split_pro_result: split_pro result
    :param cluster_object_count: prior cluster count
    :return: 
    """
    pro_result_read = dict()
    anchor = tf.identity(split_pro_result['anchor'])
    with tf.name_scope('pro_result_read'):
        y = list()
        x = list()
        h = list()
        w = list()
        for i in range(0, cluster_object_count):
            y.append(tf.expand_dims(anchor[:, :, :, i * 4 + 0], axis=-1))
            x.append(tf.expand_dims(anchor[:, :, :, i * 4 + 1], axis=-1))
            h.append(tf.expand_dims(anchor[:, :, :, i * 4 + 2], axis=-1))
            w.append(tf.expand_dims(anchor[:, :, :, i * 4 + 3], axis=-1))
        pro_result_read['y'] = tf.concat(y, axis=-1, name='y_read')
        pro_result_read['x'] = tf.concat(x, axis=-1, name='x_read')
        pro_result_read['h'] = tf.concat(h, axis=-1, name='h_read')
        pro_result_read['w'] = tf.concat(h, axis=-1, name='w_read')
        pass
    return pro_result_read
    pass


# todo:use gt_anchor and anchor_pro to calc iou， output for calc precision loss
def calc_iou(pro_result_read_result, place_process_result, scalar, prior_h, prior_w):
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


# todo: generate the loss op
def calc_loss(split_pro_result, gt_process_result, calc_iou_result):
    lambda_obj = 1.0
    lambda_noobj = 0.1
    anchor_pro = split_pro_result['anchor']
    precision_pro = split_pro_result['precision']
    class_pro = split_pro_result['class']
    p_mask = gt_process_result['p_mask']
    n_mask = gt_process_result['n_mask']
    gt_anchor = gt_process_result['anchor']
    # gt_precision = place_gt_result['precision']
    gt_class = gt_process_result['class']
    with tf.name_scope('loss'):
        with tf.name_scope('anchor_loss'):
            p_anchor_loss = tf.multiply(
                lambda_obj,
                tf.reduce_sum(
                    tf.multiply(
                        tf.square(
                            tf.subtract(
                                gt_anchor,
                                anchor_pro
                            )
                        ),
                        p_mask
                    )
                ),
                name='p_loss'
            )
            n_anchor_loss = tf.multiply(
                lambda_noobj,
                tf.reduce_sum(
                    tf.multiply(
                        tf.square(
                            tf.subtract(
                                gt_anchor,
                                anchor_pro
                            )
                        ),
                        n_mask
                    )
                ),
                name='n_loss'
            )
            anchor_loss = tf.add(p_anchor_loss, n_anchor_loss, name='loss')
            pass
        with tf.name_scope('precision_loss'):
            n_precision_loss = tf.multiply(
                tf.reduce_sum(
                    tf.multiply(
                        tf.square(
                            tf.subtract(
                                precision_pro,
                                0
                            )
                        ),
                        n_mask
                    )
                ),
                lambda_noobj,
                name='n_loss'
            )
            p_precision_loss = tf.multiply(
                tf.reduce_sum(
                    tf.multiply(
                        tf.square(
                            tf.subtract(
                                precision_pro,
                                calc_iou_result
                            )
                        ),
                        p_mask
                    )
                ),
                lambda_obj,
                name='p_loss'
            )
            precision_loss = tf.add(p_precision_loss, n_precision_loss, name='loss')
            pass
        with tf.name_scope('class_loss'):
            p_class_loss = tf.multiply(
                lambda_obj,
                tf.reduce_sum(
                    tf.multiply(
                        tf.square(
                            tf.subtract(
                                gt_class,
                                class_pro
                            )
                        ),
                        p_mask
                    )
                ),
                name='p_loss'
            )
            n_class_loss = tf.multiply(
                lambda_noobj,
                tf.reduce_sum(
                    tf.multiply(
                        tf.square(
                            tf.subtract(
                                gt_class,
                                class_pro
                            )
                        ),
                        n_mask
                    )
                ),
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
    loss, place = append_yolo2(feature_feed, 100, 3, [10, 5, 3, 4], [2, 3, 4, 8], 32)
    pass
