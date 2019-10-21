# coding=utf-8
import numpy as np
import cv2


def SemanticLabelResize(dst_size, class_amount):
    '''
    this function return a lambda function which can be use to resize the semantic label
    dst_size: [height, width]
    class_amount: the class amount include the background
    resize: the function which receive a source label and return the resized label
        the source label should be in format: [batch_size, height, width, 1] or [batch_size, height, width]
        the resized label is in format: [batch_size] + dst_size (dims:3)
    '''
    assert (type(dst_size) == list or type(dst_size) == tuple) is True

    def resize(label):
        if label.shape[-1] == 1:
            one_hot = np.eye(class_amount, dtype=np.float32)[np.squeeze(label, axis=-1).astype(np.int32)]
            pass
        else:
            one_hot = np.eye(class_amount, dtype=np.float32)[label.astype(np.int32)]
            pass
        batch_class_combine = np.reshape(np.transpose(one_hot, axes=[1, 2, 3, 0]), [one_hot.shape[1], one_hot.shape[2], one_hot.shape[0] * one_hot.shape[3]])
        resized = cv2.resize(batch_class_combine, (dst_size[1], dst_size[0]))
        back_shape = np.transpose(np.reshape(resized, list(dst_size) + [class_amount] + [one_hot.shape[0]]), [3, 0, 1, 2])
        return np.argmax(back_shape, axis=-1)
    return resize
    pass
