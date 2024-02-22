# coding=utf-8
import Putil.loger as plog
from colorama import Fore
import numpy as np

root_logger = plog.PutilLogConfig('build_function').logger()
root_logger.setLevel(plog.DEBUG)
DenseToOneHotLogger = root_logger.getChild("DensetoOneHot")
DenseToOneHotLogger.setLevel(plog.DEBUG)


# : tested
def dense_to_one_hot(array, class_num, dtype_, axis=-1):
    """
    this function offer a method to change the numpy array to one hot like base on axis
    as we know array dims_size in axis should be 1
    keep_shape
    :param array: a numpy array, data type should be int
    :param class_num: one hot class number
    :param dtype_: numpy data type declaration
    :param axis: broadcast dim
    :return:
    '
    algorithm:
        base on axis: base_point(the local of the dim we want to one hot)
        we transpose the array to [...., base_point]
        and than we make a zeros array [array_element_amount, class_num]
        make an array np.arange(num_labels) * class_num for support the offset
        which means the step to make sure the array.flat location which set to 1(dtype_)
    '
    """
    array_shape = array.shape
    assert array_shape[axis] == 1, DenseToOneHotLogger.error(Fore.RED + 'dim {0} should be size: 1'.format(axis))
    if array.max() >= class_num:
        raise ValueError('class_num(a) should bigger than the max of array(b), '
                         'but a vs. b = {0} vs.{1}'.format(class_num, array.max()))
    base_point = axis % len(array_shape)
    transpose_axes = []
    back_transpose_axes = []
    DenseToOneHotLogger.debug("start generate transpose_axes and back_transpose_axes")
    if base_point == len(array_shape):
        pass
    elif base_point == 0:
        transpose_axes += list(range(1, len(array_shape)))
        transpose_axes.append(0)
        back_transpose_axes += [len(array_shape) - 1] + list(range(1, len(array_shape)))
        pass
    else:
        f_start = 0
        f_end = base_point
        b_start = base_point + 1
        b_end = len(array_shape) - 1
        transpose_axes += list(range(f_start, f_end))
        transpose_axes += list(range(b_start, b_end))
        transpose_axes.append(base_point)
        back_transpose_axes += list(range(f_start, base_point))
        back_transpose_axes += [len(array_shape) - 1]
        back_transpose_axes += list(range(base_point, len(array_shape) - 2))
    DenseToOneHotLogger.debug('transpose')
    np.transpose(array, transpose_axes)
    shape = list(array.shape)
    shape[-1] = class_num
    num_labels = 1
    for i in list(np.transpose(array).shape)[0:]:
        num_labels *= i
        pass

    index_offset = np.arange(num_labels) * class_num
    label_one_hot = np.zeros(shape, dtype=dtype_)
    label_one_hot.flat[index_offset + array.ravel()] = 1
    DenseToOneHotLogger.debug("re transpose")
    np.transpose(label_one_hot, back_transpose_axes)
    return label_one_hot
    pass


# : test more
def __test_dense_to_one_hot(t_logger):
    logger = t_logger.getChild("test_dense_to_one_hot")
    logger.debug(Fore.GREEN + 'start test 3-d')
    test_data_one = np.zeros([3, 3, 1], np.int64)
    test_data_one[1, 1] = 2
    test_data_one[0, 2] = 0
    test_data_one[2, 0] = 1
    logger.debug('test data : {0}'.format(test_data_one))
    one_done = dense_to_one_hot(test_data_one, 3, np.float32, -1)
    try:
        assert False not in (one_done[1, 1, :] == np.array([0, 0, 1])), \
            logger.debug(Fore.RED + '{0} should be: {1}, but: {2}'.format(
                '[1, 1, :]', [0, 0, 1], one_done[1, 1, :]))
        assert False not in (one_done[0, 2, :] == np.array([1, 0, 0])), \
            logger.debug(Fore.RED + '{0} should be: {1}, but: {2}'.format(
                '[1, 1, :]', [1, 0, 0], one_done[0, 2, :]))
        assert False not in (one_done[2, 0, :] == np.array([0, 1, 0])), \
            logger.debug(Fore.RED + '{0} should be: {1}, but: {2}'.format(
                '[1, 1, :]', [0, 1, 0], one_done[2, 0, :]))
    except AssertionError:
        logger.debug(Fore.RED + 'test 3-d faild')
        return False
    logger.debug(Fore.GREEN + 'start test 3-d')
    return True
    pass


if __name__ == '__main__':
    from optparse import OptionParser
    import logging
    parser = OptionParser(usage='usage:%prog [options] arg1 arg2')
    parser.add_option(
        '--level',
        action='store',
        dest='Level',
        default='Info',
        help='test app only use stream ,set the log label'
             'default : info'
    )
    (options, args) = parser.parse_args()
    plog.PutilLogConfig.config_handler(plog.stream_method)
    plog.PutilLogConfig.config_log_level(stream=plog.LogReflect(options.Level).Level)
    logger = plog.PutilLogConfig('T_build_function').logger()
    logger.setLevel(logging.DEBUG)
    logger.info('>>>>>>test the __2d_dense_to_one_hot>>>>>>')
    if __test_dense_to_one_hot(logger) is True:
        logger.info('------------successful------------')
    else:
        logger.info('------------failed------------')
    pass
