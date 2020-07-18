# coding=utf-8
import numpy as np

def rect_angle_over_border(bboxes, width, height):
    bboxes = np.array(bboxes)
    flag = (np.argwhere(bboxes[:, 0: 2] < 0).shape[0] == 0) & \
        (np.argwhere((bboxes[:, 0] + bboxes[:, 2]) > width).shape[0] == 0) & \
            (np.argwhere((bboxes[:, 1] + bboxes[: ,3]) > height).shape[0] == 0)
    return not flag


def clip_box(bboxes, width, height):
    '''
     @brief clip the box to avoid border cross
     @note 
     the bbox which is over border would be replaced by [None, None, None, None]
     @param[in] width
     @param[in] height
     @param[in] bboxes list [[x, y, width, height]]
    '''
##In[]:
#    import numpy as np
#    width = 80 
#    height = 80
#    bboxes = np.reshape(np.random.sample(40), [10, 4]) * 100
    bboxes = np.array(bboxes)
    bboxes[:, 2: 4] = bboxes[:, 0: 2] + bboxes[:, 2: 4]
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, width - 1)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, height - 1)
    bboxes[:, 2: 4] = bboxes[:, 2: 4] - bboxes[:, 0: 2]
    #bboxes = np.delete(bboxes, np.argwhere(bboxes[:, 2: 4] <= 0)[:, 0: 1], axis=0)
#    print(bboxes)
    bboxes[np.argwhere(bboxes[:, 2: 4] <= 0)[:, 0], :] = [None, None, None, None]
#    print(bboxes)
    return bboxes.tolist()
##In[]
##In[]


def clip_box_using_image(bboxes, image):
    '''
     @brief clip the box to avoid border cross
     @param[in] image [height, width[, channel]]
     @param[in] bboxes list [[x, y, width, height]]
    '''
    return clip_box(bboxes, image.shape[1], image.shape[0])