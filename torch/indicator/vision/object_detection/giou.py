# coding=utf-8

import torch


def compute_iou_giou(pre, gt):
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
    x1, y1, width, height = torch.split(pre, 1, dim=-1)
    x1g, y1g, widthgt, heightgt = torch.split(gt, 1, dim=-1)
    x2, y2 = (x1 + width, y1 + height)
    x2g, y2g = (x1g + widthgt, y1g + heightgt)

    cap_x1 = torch.max(x1, x1g)
    cap_y1 = torch.max(y1, y1g)
    cap_x2 = torch.min(x2, x2g)
    cap_y2 = torch.min(y2, y2g)

    closure_x1 = torch.min(x1, x1g)
    closure_y1 = torch.min(y1, y1g)
    closure_x2 = torch.max(x2, x2g)
    closure_y2 = torch.max(y2, y2g)

    cap = torch.zeros(x1.shape).to(pre)
    mask = (cap_y2 > cap_y1) * (cap_x2 > cap_x1)
    cap[mask] = (cap_x2[mask] - cap_x1[mask]) * (cap_y2[mask] - cap_y1[mask]) # cap 
    cup = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g) - cap + 1e-32 # pre \cup gtr

    iou = cap / cup # whild the cap is zero , there would be no grad backpropagation(take a look at where the cap come from)

    area_c = (closure_x2 - closure_x1) * (closure_y2 - closure_y1) + 1e-7
    giou = iou - ((area_c - cup) / area_c)
    return iou, giou