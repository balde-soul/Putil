# coding=utf-8

import torch 


def compute_diou(pre, gt):
    '''
     @brief
     @note
     https://arxiv.org/pdf/1911.08287.pdf
     DIoU = IoU - \
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

    center_x = (x1 + x2) / 2 
    center_y = (y1 + y2) / 2 
    center_xgt = (x1g + x2g) / 2
    center_ygt = (y1g + y2g) / 2

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

    inter_diag = (center_x - center_xgt)**2 + (center_y - center_ygt)**2

    outer_diag = (closure_x1 - closure_x2) ** 2 + (closure_y1 - closure_y2) ** 2 + 1e-32

    dious = iou - (inter_diag) / outer_diag

    return iou, dious