# coding=utf-8
import Putil.data.base_operation as bo
import numpy as np


def test_SemanticLabelResize():
    batch_size = 1
    src_height = 500
    src_width = 600
    class_amount = 5
    dst_size = [200, 250]
    src = np.zeros([batch_size, src_height, src_width, 1], dtype=np.int32)
    src[0: 100, 0: 150] = 1
    src[100: 200, 100: 250] = 2
    src[200: 300, 200: 350] = 3
    src[300: 400, 300: 450] = 4
    func = bo.SemanticLabelResize(dst_size, class_amount)
    dst = func(src)
    assert dst.shape == tuple([batch_size] + dst_size), print(dst.shape)
    pass


if __name__ == '__main__':
    test_SemanticLabelResize()
    pass
