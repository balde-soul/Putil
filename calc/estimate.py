# coding = utf-8
import numpy as np
from colorama import init, Fore, Back, Style


def __iou_chw(rect1, rect2):
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
