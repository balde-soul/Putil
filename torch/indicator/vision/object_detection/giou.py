# coding=utf-8

import torch


def compute_iou_giou(pre, gt, transform_weights=None):
    '''
     @brief
     @note
     https://arxiv.org/pdf/1902.09630.pdf
     iou在iou为0时无法直接优化，针对bbox的回归无法等价于iou的回归，提出giou可作为目标进行优化
     @param[in] pre
     float or double, positivated, [batch_size, ..., one_box] content of box: (top_left_x, top_left_y, width, height)
     @prarm[in] gt 
     float or double, positivated, [batch_size, ..., one_box] content of box: (top_left_x, top_left_y, width, height)
     @ret
     0: the iou [batch_size, ..., 1]
     1: the giou [batch_size, ..., 1]
    '''
    if transform_weights is None:
        transform_weights = (1., 1., 1., 1.)

    batch_size = pre.size(0)

    x1, y1, width, height = torch.split(pre, 1, dim=-1)
    x1g, y1g, widthgt, heightgt = torch.split(gt, 1, dim=-1)
    x2, y2 = (x1 + width, y1 + height)
    x2g, y2g = (x1g + widthgt, y1g + heightgt)

    xkis1 = torch.max(x1, x1g)
    ykis1 = torch.max(y1, y1g)
    xkis2 = torch.min(x2, x2g)
    ykis2 = torch.min(y2, y2g)

    xc1 = torch.min(x1, x1g)
    yc1 = torch.min(y1, y1g)
    xc2 = torch.max(x2, x2g)
    yc2 = torch.max(y2, y2g)

    cap = torch.zeros(x1.shape).to(pre)
    mask = (ykis2 > ykis1) * (xkis2 > xkis1)
    cap[mask] = (xkis2[mask] - xkis1[mask]) * (ykis2[mask] - ykis1[mask]) # cap 
    cup = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - cap + 1e-7 # pre \cup gtr

    iou = cap / cup # whild the cap is zero , there would be no grad backpropagation(take a look at where the cap come from)

    area_c = (xc2 - xc1) * (yc2 - yc1) + 1e-7
    giou = iou - ((area_c - cup) / area_c)
    return iou, giou