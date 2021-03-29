# coding=utf-8
import torch
from torch import nn


def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


##@brief 解码center_net backend的输出，生成box，class，score，center_xyes
# @note
class CenterNetDecode(torch.nn.Module):
    def __init__(self, downsample_rate, threshold, nms_range=3):
        torch.nn.Module.__init__(self)
        self._downsample_rate = downsample_rate
        self._threshold = threshold
        self._nms_range = nms_range
        pass

    ##@brief
    # @note
    # @param[in] radiace_factor [batch, class_amount, height, width] the predict class score
    # @param[in] pre_box [batch, 4, height, width] the predict box (x_shift, y_shift, w_fit, h_fit) in the data
    # @param[in] sample_rate
    # @param[in] threshold
    # @ret
    # [0] boxes list [tensor([[top_left_x, top_left_y, width, height], ...]), ...] shape: [[4]]
    # [1] classes list [tensor([classes_index, ...]), ...] shape: [[1]]
    # [2] score tensor() shape: [batch, 1, heiht, width]
    # [3] center_xyes list [tensor([[center_x, center_y]]), ...]
    def forward(self, radiace_factor, box):
        batch, cat, height, width = radiace_factor.size()
        radiace_factor = _nms(radiace_factor, self._nms_range)
        score, sinds = radiace_factor.max(dim=1, keepdim=True)
        sinds = score.gt(self._threshold)
        boxes = [] # (top_left_xy wh)
        classes = []
        inds_indexes = []
        center_xyes = []
        result = []
        for b in range(0, batch):
            #inds = score[b, 0, :].gt(threshold)
            inds = sinds[b, 0, ...]
            filter_box = box[b, :, inds].permute(1, 0).contiguous()
            if filter_box.shape[0] == 0:
                result.append(torch.BoolTensor([False]))
                result.append(torch.BoolTensor([False]))
                result.append(torch.BoolTensor([False]))
                result.append(torch.BoolTensor([False]))
                continue
            inds_index = torch.where(inds)
            inds_indexes.append(self._sample_rate * inds_index)
            center_xy = torch.stack(torch.where(inds)[::-1], dim=-1) * self._sample_rate + filter_box[..., 0: 2] * self._sample_rate
            #print('center_xy: {}'.format(center_xy))
            wh = filter_box[..., 2: 4] * self._sample_rate
            #print('wh: {}'.format(wh))
            x1y1 = center_xy - wh * 0.5
            x1y1 = torch.stack([torch.clip(x1y1[:, 0], 0.0, width * self._sample_rate), torch.clip(x1y1[:, 1], 0.0, height * self._sample_rate)], axis=-1)
            x2y2 = center_xy + wh * 0.5
            x2y2 = torch.stack([torch.clip(x2y2[:, 0], 0.0, width * self._sample_rate), torch.clip(x2y2[:, 1], 0.0, height * self._sample_rate)], axis=-1)
            wh = x2y2 - x1y1
            assert (wh.lt(0.0).sum() == 0).item()
            # the box
            result.append(torch.cat([x1y1, wh], dim=-1))
            # the class
            result.append(inds)
            # the score
            result.append(score[b, 0, inds_index[0], inds_index[1]])
            # the center
            result.append(center_xy)
        return result