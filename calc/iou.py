# coding = utf-8
import numpy as np
from colorama import init, Fore, Back, Style


"""
Summary:
    IoU Calculate:
        the IoU between Box_1 and Box_2:
             F(x): if x < 0 , =0; else, =x;
        IoU: F(max_top - min_bottom) * F(min_right - max_left)
             max_top = max(Box_1_top_y, Box_2_top_y)
             min_bottom = min(Box_1_bottom_y, Box_2_bottom_y)
             min_right = min(Box_1_right_x, Box_2_right_x)
             max_left = max(Box_1_left_x, Box_2_left_x)
             base on different parameter , generate different way to calculate the max_top, min_bottom, min_right and
             max_left
              max_top    
                ===     
                 |
             ----|--
   Box_1<---|    |  |--->min_right
            |   ----|-----------
     IoU<---|--|////|           |
max_left<---|--|////|           |
            |  |////|           |
             --|----            |
             | |                |
             | |                |-----> Box_2
            ===|                |
      min_bottom----------------
"""


def __iou_chw(rect1, rect2):
    """
    calculate the IoU between rect1 and rect2, use the [center_y, center_x, height, width]
    :param rect1:
    :param rect2:
    :return:
    """
    y1, x1, h1, w1 = rect1
    y2, x2, h2, w2 = rect2
    if (abs(x1 - x2) < ((w1 + w2) / 2.0)) and (abs(y1 - y2) < ((h1 + h2) / 2.0)):
        left = max((x1 - (w1 / 2.0)), (x2 - (w2 / 2.0)))
        upper = max((y1 - (h1 / 2.0)), (y2 - (h2 / 2.0)))

        right = min((x1 + (w1 / 2.0)), (x2 + (w2 / 2.0)))
        bottom = min((y1 + (h1 / 2.0)), (y2 + (h2 / 2.0)))

        inter_w = abs(left - right)
        inter_h = abs(upper - bottom)
        inter_square = inter_w * inter_h
        union_square = (w1 * h1) + (w2 * h2) - inter_square

        iou = inter_square / union_square * 1.0
        inter_rect = [(upper + bottom) * 0.5, (left + right) * 0.5, bottom - upper, right - left]
    else:
        iou = 0
        inter_rect = [None, None, None, None]
        pass
    return iou, inter_rect
    pass


def __to_chw(*rects, **options):
    TP = options.pop('TP', False)
    LHW = options.pop('LHW', False)
    CHW = options.pop('CHW', False)
    assert np.count_nonzero([TP, LHW, CHW]) == 1, \
        'TP, LHW, CHW should have only one True, but {0}'.format(np.count_nonzero([TP, LHW, CHW]))
    assert len(rects) >= 1, 'no input rect'
    get = []
    if TP:
        [get.append([(i[0] + i[2]) * 0.5, (i[1] + i[3]) * 0.5, i[2] - i[0], i[3] - i[1]]) for i in rects]
        return get
    if LHW:
        [get.append([i[0] + 0.5 * i[2], i[1] + 0.5 * i[3], i[2], i[3]]) for i in rects]
        return get
    if CHW:
        return rects
    pass


def calc_iou(*rects, **options):
    """
    多个rects计算iou存在错误
    计算一组rects的iou
    :param rects: 一组rects
    :param options: 
        :keyword TP : rects使用两点（左上， 右下）表示 [left_y, left_x, right_y, right_x]
        :keyword LHW : rects使用左上与高宽表示 [left_y, left_x, height, width]
        :keyword CHW : rects使用中心点与高宽表示 [center_y, center_x, height, width]
    :return: 
    """
    # fixme:多个rects计算iou存在error
    TP = options.pop('TP', False)
    LHW = options.pop('LHW', False)
    CHW = options.pop('CHW', False)

    rects = __to_chw(*rects, TP=TP, LHW=LHW, CHW=CHW)

    inter_rect = rects[0]
    iou = None
    for i in range(1, len(rects)):
        iou, inter_rect_new = __iou_chw(inter_rect, rect2=rects[i])
        if None in inter_rect_new:
            return iou
        else:
            inter_rect = inter_rect_new
    return iou
    pass


"""
Implement calc_iou_matrix_thw:
    base on center_y_x and height width, there is algorithm:
        max_top: max(-0.5 * group1_h, group_2_y - 0.5 * group2_h)
        min_bottom: min(0.5 * group1_h, group_2_y + 0.5 * group2_h)
        min_right: min(0.5 * group1_w, group2_x + 0.5 * group2_w)
        max_left: min(-0.5 * group1_w, group2_x - 0.5 * group2_w)

        use[[center_y, center_x, height, width], ....] as an example:
            in order to create the IoU matrix
            we should create group1_Box_M IoU group2_Box_N

            we make group1 data repeat n cross row
            just like:
                -0.5 * group1_h: 
                [[group1_box_1_top_y, ..n.., group1_box_1_top_y], 
                 [group1_box_2_top_y, ..n.., group1_box_2_top_y],
                  :
                  m
                  :,
                 [group1_box_m_top_y, ..n.., group1_box_m_top_y],
                ]
            we make group2 data repeat m cross col
                and group2 just make more one process transpose
            and then use the algorithm
            get then max_top, min_bottom, min_right, max_left Matrix
            and then make element which lower than zeros zeroed
            finally generate a m x n IoU matrix
"""


def calc_iou_matrix_ohw(
        group1,
        group2,
        group1_h_index=2,
        group1_w_index=3,
        group2_y_index=0,
        group2_x_index=1,
        group2_h_index=2,
        group2_w_index=3
):
    """
    this function is for standard group1 IoU random group2
    which means that the box in the group1 have the same center_y_x, and group2 carry the data
    [offset_y, offset_x, height, width]， offset means the offset pixel to the standard box center
    calculate the IoU matrix base on group1 and group2 which carry the parameter top_y, top_x, height and width
    :param group1: [[height, width], ....] according to  default group1_*_index
    :param group2: [[offset_y, offset_x, height, width], ...] according to default group2_*_index
    :param group1_h_index: parameter represent the index of h in group1
    :param group1_w_index: parameter represent the index of 2 in group1
    :param group2_y_index: parameter represent the index of y in group2
    :param group2_x_index: parameter represent the index of x in group2
    :param group2_h_index: parameter represent the index of h in group2
    :param group2_w_index: parameter represent the index of w in group2
    :return:
        group1_box_0 iou group2_box_0, group1_box_0 iou group2_box_1, ..., group1_box_0 iou group2_box_(n - 1), group1_box_0 iou group2_box_n
                      ,                              ,                                   ,                                    ,
        group1_box_1 iou group2_box_0,              ...             , ...,              ...                   , group1_box_1 iou group2_box_n
                      ,
                     ...                            ...
                      ,
                     ...                                              ...
                      ,
                     ...                                                                ...
                      ,
        group1_box_m iou group2_box_0,              ...             , ...,              ...                   , group1_box_m iou group2_box_n
    """
    g_1_matrix = np.array(group1)
    g_2_matrix = np.array(group2)
    group_1_amount = len(g_1_matrix)
    group_2_amount = len(g_2_matrix)
    g_1_area_cross_row = (g_1_matrix[:, group1_h_index] * g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_2_area_cross_col = (g_2_matrix[:, group2_h_index] * g_2_matrix[:, group2_w_index]).repeat(
        group_1_amount).reshape(group_2_amount, group_1_amount).T
    g_1_top_y_matrix_cross_row = (-0.5 * g_1_matrix[:, group1_h_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_bottom_y_matrix_cross_row = (0.5 * g_1_matrix[:, group1_h_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_right_x_matrix_cross_row = (0.5 * g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_left_x_matrix_cross_row = (-0.5 * g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_2_top_y_matrix_cross_col = (g_2_matrix[:, group2_y_index] - 0.5 * g_2_matrix[:, group2_h_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    g_2_bottom_y_matrix_cross_col = (g_2_matrix[:, group2_y_index] + 0.5 * g_2_matrix[:, group2_h_index]).repeat(
        group_1_amount).reshape(group_2_amount, group_1_amount).T
    g_2_right_x_matrix_cross_col = (g_2_matrix[:, group2_x_index] + 0.5 * g_2_matrix[:, group2_w_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    g_2_left_x_matrix_cross_col = (g_2_matrix[:, group2_x_index] - 0.5 * g_2_matrix[:, group2_x_index]).repeat(
        group_1_amount).reshape(group_2_amount, group_1_amount).T
    # calculate the overlap box
    max_top = np.max(np.concatenate((np.expand_dims(g_1_top_y_matrix_cross_row, -1),
                                        np.expand_dims(g_2_top_y_matrix_cross_col, -1)), -1), -1)
    min_bottom = np.min(np.concatenate((np.expand_dims(g_1_bottom_y_matrix_cross_row, -1),
                                     np.expand_dims(g_2_bottom_y_matrix_cross_col, -1)), -1), -1)
    min_right = np.min(np.concatenate((np.expand_dims(g_1_right_x_matrix_cross_row, -1),
                                       np.expand_dims(g_2_right_x_matrix_cross_col, -1)), -1), -1)
    max_left = np.max(np.concatenate((np.expand_dims(g_1_left_x_matrix_cross_row, -1),
                                      np.expand_dims(g_2_left_x_matrix_cross_col, -1)), -1), -1)
    # calculate cross area
    crossed_height = min_bottom - max_top
    crossed_width = min_right - max_left
    # apply ReLU
    crossed_height[crossed_height < 0] = 0
    crossed_width[crossed_width < 0] = 0
    iou_area = crossed_height * crossed_width
    iou = iou_area / (g_1_area_cross_row + g_2_area_cross_col - iou_area)
    return iou
    pass


"""
Implement calc_iou_matrix_thw:
    base on center_y_x and height width, there is algorithm:
        max_top: max(group1_y, group_2_y)
        min_bottom: min(group1_y + group1_h, group_2_y + group2_h)
        min_right: min(group_1_x + group1_w, group2_x + group2_w)
        max_left: min(group_1_x, group2_x)
        
        use[[center_y, center_x, height, width], ....] as an example:
            in order to create the IoU matrix
            we should create group1_Box_M IoU group2_Box_N
            
            we make group1 data repeat n cross row
            just like:
                group1_y: 
                [[group1_box_1_top_y, ..n.., group1_box_1_top_y], 
                 [group1_box_2_top_y, ..n.., group1_box_2_top_y],
                  :
                  m
                  :,
                 [group1_box_m_top_y, ..n.., group1_box_m_top_y],
                ]
            we make group2 data repeat m cross col
                and group2 just make more one process transpose
            and then use the algorithm
            get then max_top, min_bottom, min_right, max_left Matrix
            and then make element which lower than zeros zeroed
            finally generate a m x n IoU matrix
"""


def calc_iou_matrix_thw(
        group1,
        group2,
        group1_y_index=0,
        group1_x_index=1,
        group1_h_index=2,
        group1_w_index=3,
        group2_y_index=0,
        group2_x_index=1,
        group2_h_index=2,
        group2_w_index=3
):
    """
    calculate the IoU matrix base on group1 and group2 which carry the parameter top_y, top_x, height and width
    :param group1: [[top_y, top_x, height, width], ....] according to  default group1_*_index
    :param group2: [[top_y, top_x, height, width], ...] according to default group2_*_index
    :param group1_y_index: parameter represent the index of y in group1
    :param group1_x_index: parameter represent the index of x in group1
    :param group1_h_index: parameter represent the index of h in group1
    :param group1_w_index: parameter represent the index of 2 in group1
    :param group2_y_index: parameter represent the index of y in group2
    :param group2_x_index: parameter represent the index of x in group2
    :param group2_h_index: parameter represent the index of h in group2
    :param group2_w_index: parameter represent the index of w in group2
    :return:
        group1_box_0 iou group2_box_0, group1_box_0 iou group2_box_1, ..., group1_box_0 iou group2_box_(n - 1), group1_box_0 iou group2_box_n
                      ,                              ,                                   ,                                    ,
        group1_box_1 iou group2_box_0,              ...             , ...,              ...                   , group1_box_1 iou group2_box_n
                      ,
                     ...                            ...
                      ,
                     ...                                              ...
                      ,
                     ...                                                                ...
                      ,
        group1_box_m iou group2_box_0,              ...             , ...,              ...                   , group1_box_m iou group2_box_n
    """
    g_1_matrix = np.array(group1)
    g_2_matrix = np.array(group2)
    group_1_amount = len(g_1_matrix)
    group_2_amount = len(g_2_matrix)
    g_1_area_cross_row = (g_1_matrix[:, group1_h_index] * g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_2_area_cross_col = (g_2_matrix[:, group2_h_index] * g_2_matrix[:, group2_w_index]).repeat(
        group_1_amount).reshape(group_2_amount, group_1_amount).T
    g_1_bottom_y_matrix_cross_row = (g_1_matrix[:, group1_y_index] + g_1_matrix[:, group1_h_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_top_y_matrix_cross_row = (g_1_matrix[:, group1_y_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_right_x_matrix_cross_row = (g_1_matrix[:, group1_x_index] + g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_left_x_matrix_cross_row = (g_1_matrix[:, group1_x_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_2_bottom_y_matrix_cross_col = (g_2_matrix[:, group2_y_index] + g_2_matrix[:, group2_h_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    g_2_top_y_matrix_cross_col = (g_2_matrix[:, group2_y_index]).repeat(
        group_1_amount).reshape(group_2_amount, group_1_amount).T
    g_2_right_x_matrix_cross_col = (g_2_matrix[:, group2_x_index] + g_2_matrix[:, group2_w_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    g_2_left_x_matrix_cross_col = (g_2_matrix[:, group2_x_index]).repeat(
        group_1_amount).reshape(group_2_amount, group_1_amount).T
    # calculate the overlap box
    min_bottom = np.min(
        np.concatenate(
            (np.expand_dims(g_1_bottom_y_matrix_cross_row, -1),
             np.expand_dims(g_2_bottom_y_matrix_cross_col, -1)),
            -1
        ),
        -1
    )
    max_top = np.max(
        np.concatenate(
            (np.expand_dims(g_1_top_y_matrix_cross_row, -1),
             np.expand_dims(g_2_top_y_matrix_cross_col, -1)),
            -1
        ),
        -1
    )
    min_right = np.min(
        np.concatenate(
            (np.expand_dims(g_1_right_x_matrix_cross_row, -1),
             np.expand_dims(g_2_right_x_matrix_cross_col, -1)),
            -1
        ),
        -1
    )
    max_left = np.max(
        np.concatenate(
            (np.expand_dims(g_1_left_x_matrix_cross_row, -1),
             np.expand_dims(g_2_left_x_matrix_cross_col, -1)),
            -1
        ),
        -1
    )
    # calculate cross area
    crossed_height = min_bottom - max_top
    crossed_width = min_right - max_left
    # apply ReLU
    crossed_height[crossed_height < 0] = 0
    crossed_width[crossed_width < 0] = 0
    iou_area = crossed_height * crossed_width
    iou = iou_area / (g_1_area_cross_row + g_2_area_cross_col - iou_area)
    return iou
    pass


"""
Implement calc_iou_matrix_chw:
    base on center_y_x and height width, there is algorithm:
        max_top: max(group1_y - 0.5 * group1_h, group_2_y - 0.5 * group2_h)
        min_bottom: min(group1_y + 0.5 * group1_h, group_2_y + 0.5 * group2_h)
        min_right: min(group_1_x + 0.5 * group1_w, group2_x + 0.5 * group2_w)
        max_left: min(group_1_x - 0.5 * group1_w, group2_x - 0.5 * group2_w)
        
        use[[center_y, center_x, height, width], ....] as an example:
            in order to create the IoU matrix
            we should create group1_Box_M IoU group2_Box_N
            
            we make group1 data repeat n cross row
            just like:
                group1_y - 0.5 * group1_h: 
                [[group1_box_1_top_y, ..n.., group1_box_1_top_y], 
                 [group1_box_2_top_y, ..n.., group1_box_2_top_y],
                  :
                  m
                  :,
                 [group1_box_m_top_y, ..n.., group1_box_m_top_y],
                ]
            we make group2 data repeat m cross col
                and group2 just make more one process transpose
            and then use the algorithm
            get then max_top, min_bottom, min_right, max_left Matrix
            and then make element which lower than zeros zeroed
            finally generate a m x n IoU matrix
"""


def calc_iou_matrix_chw(
        group1,
        group2,
        group1_y_index=0,
        group1_x_index=1,
        group1_h_index=2,
        group1_w_index=3,
        group2_y_index=0,
        group2_x_index=1,
        group2_h_index=2,
        group2_w_index=3
):
    """
    calculate the IoU matrix base on group1 and group2 which carry the parameter center_y, center_x, height and width
    :param group1: [[center_y, center_x, height, width], ....] according to  default group1_*_index
    :param group2: [[center_y, center_x, height, width], ...] according to default group2_*_index
    :param group1_y_index: parameter represent the index of y in group1
    :param group1_x_index: parameter represent the index of x in group1
    :param group1_h_index: parameter represent the index of h in group1
    :param group1_w_index: parameter represent the index of 2 in group1
    :param group2_y_index: parameter represent the index of y in group2
    :param group2_x_index: parameter represent the index of x in group2
    :param group2_h_index: parameter represent the index of h in group2
    :param group2_w_index: parameter represent the index of w in group2
    :return:
        group1_box_0 iou group2_box_0, group1_box_0 iou group2_box_1, ..., group1_box_0 iou group2_box_(n - 1), group1_box_0 iou group2_box_n
                      ,                              ,                                   ,                                    ,
        group1_box_1 iou group2_box_0,              ...             , ...,              ...                   , group1_box_1 iou group2_box_n
                      ,
                     ...                            ...
                      ,
                     ...                                              ...
                      ,
                     ...                                                                ...
                      ,
        group1_box_m iou group2_box_0,              ...             , ...,              ...                   , group1_box_m iou group2_box_n
    """
    g_1_matrix = np.array(group1)
    g_2_matrix = np.array(group2)
    group_1_amount = len(g_1_matrix)
    group_2_amount = len(g_2_matrix)
    g_1_area_cross_row = (g_1_matrix[:, group1_h_index] * g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_2_area_cross_col = (g_2_matrix[:, group2_h_index] * g_2_matrix[:, group2_w_index]).repeat(
        group_1_amount).reshape(group_2_amount, group_1_amount).T
    g_1_bottom_y_matrix_cross_row = (g_1_matrix[:, group1_y_index] + 0.5 * g_1_matrix[:, group1_h_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_top_y_matrix_cross_row = (g_1_matrix[:, group1_y_index] - 0.5 * g_1_matrix[:, group1_h_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_right_x_matrix_cross_row = (g_1_matrix[:, group1_x_index] + 0.5 * g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_1_left_x_matrix_cross_row = (g_1_matrix[:, group1_x_index] - 0.5 * g_1_matrix[:, group1_w_index]).repeat(
        group_2_amount).reshape([group_1_amount, group_2_amount])
    g_2_bottom_y_matrix_cross_col = (g_2_matrix[:, group2_y_index] + 0.5 * g_2_matrix[:, group2_h_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    g_2_top_y_matrix_cross_col = (g_2_matrix[:, group2_y_index] - 0.5 * g_2_matrix[:, group2_h_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    g_2_right_x_matrix_cross_col = (g_2_matrix[:, group2_x_index] + 0.5 * g_2_matrix[:, group2_w_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    g_2_left_x_matrix_cross_col = (g_2_matrix[:, group2_x_index] - 0.5 * g_2_matrix[:, group2_w_index]).repeat(
        group_1_amount).reshape([group_2_amount, group_1_amount]).T
    # calculate the overlap box
    min_bottom = np.min(np.concatenate((np.expand_dims(g_1_bottom_y_matrix_cross_row, -1), np.expand_dims(g_2_bottom_y_matrix_cross_col, -1)), -1), -1)
    max_top = np.max(np.concatenate((np.expand_dims(g_1_top_y_matrix_cross_row, -1), np.expand_dims(g_2_top_y_matrix_cross_col, -1)), -1), -1)
    min_right = np.min(np.concatenate((np.expand_dims(g_1_right_x_matrix_cross_row, -1), np.expand_dims(g_2_right_x_matrix_cross_col, -1)), -1), -1)
    max_left = np.max(np.concatenate((np.expand_dims(g_1_left_x_matrix_cross_row, -1), np.expand_dims(g_2_left_x_matrix_cross_col, -1)), -1), -1)
    # calculate cross area
    crossed_height = min_bottom - max_top
    crossed_width = min_right - max_left
    # apply ReLU
    crossed_height[crossed_height < 0] = 0
    crossed_width[crossed_width < 0] = 0
    iou_area = crossed_height * crossed_width
    iou = iou_area / (g_1_area_cross_row + g_2_area_cross_col - iou_area)
    return iou
    pass


def __test_calc_iou():
    rect1 = [0, 0, 10, 10]
    rect2 = [5, 5, 10, 10]
    rect3 = [75, 75, 100, 100]
    iou = calc_iou(rect1, rect2, LHW=True)
    try:
        assert iou == 25.0 / 175
        return True
    except Exception:
        print(Fore.RED + '>>:should be {0}, but {1}'.format(25.0 / 175, iou))
        return False
    pass


if __name__ == '__main__':
    # todo:test iou calc
    try:
        assert __test_calc_iou()
        print(Fore.GREEN + '-------------test_calc_iou: pass-------------')
    except Exception:
        print(Fore.RED + '-------------test_calc_iou: failed-------------')
        pass
pass
