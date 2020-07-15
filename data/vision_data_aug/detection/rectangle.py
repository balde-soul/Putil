# coding=utf-8

import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import Putil.data.aug as pAug
from Putil.data.vision_common_convert.bbox_convertor import BBoxConvertToCenterBox
from Putil.data.vision_common_convert.bbox_convertor import BBoxToBBoxTranslator
from Putil.data.vision_data_aug.image_aug import Resample as IR
from Putil.data.vision_data_aug.image_aug import HorizontalFlip as IH
from Putil.data.vision_data_aug.image_aug import Translate as IT
from Putil.data.vision_data_aug.image_aug import Rotate as IRE
from Putil.data.vision_data_aug.image_aug import rotate_im as rotate_im
from Putil.data.vision_data_aug.image_aug import Shear as IS


def clip_box(bboxes, image):
    bboxes = np.delete(bboxes, np.argwhere(bboxes[:, 0] > (image.shape[1] - 1)), axis=0)
    bboxes = np.delete(bboxes, np.argwhere(bboxes[:, 1] > (image.shape[0] - 1)), axis=0)
    bboxes[:, 2] = np.min([bboxes[:, 2], image.shape[1] - 1 - bboxes[:, 0]], axis=0)
    bboxes[:, 3] = np.min([bboxes[:, 3], image.shape[0] - 1 - bboxes[:, 1]], axis=0)
    return bboxes

class HorizontalFlip(IH):
    '''
     @brief aug the bboxes
     @note
     @ret
    '''
    def __init__(self):
        IH.__init__(self)
        pass

    def __call__(self, *args):
        '''
         @brief 
         @param[in] args
         [0] image [height, width, channel]
         [1] bboxes list format LTWHCR
         @ret bboxes list format LTWHCR
        '''
        image = args[0]
        bboxes = args[1]
        bboxes = np.array(bboxes)
        bboxes[:, 0] = image.shape[1] - 1 - bboxes[:, 0] - bboxes[:, 2]
        return bboxes.tolist(), 

class CombineHorizontalFlip(pAug.AugFunc):
    def __init__(self):
        self._image_aug = IH()
        self._bboxes_aug = HorizontalFlip()
        pass

    def __call__(self, *args):
        image = args[0]
        bboxes = args[1]

        img,  = self._image_aug(image)
        bboxes,  = self._bboxes_aug(image, bboxes)
        return img, bboxes

    @property
    def doc(self):
        return 'flip the image follow the horizon'
    @property
    def name(self):
        return 'HorizontalFlip'

class Resample(IR):
    def __init__(self, diff = False):
        IR.__init__(self)
        pass

    def __call__(self, *args):
        '''
         @brief
         @note
         @param[in] args tuple point *args
          args[0] image
         the image with format [height, widht[, channel]]
          args[1] bboxes 
         a list which contain the bound boxes in format LTWHCR with list
         @ret 
         a list which contain the bound boxes in format LTWHCR with list
        '''
        assert (self._resample_scale_x is not None) and (self._resample_scale_y is not None)
        image = args[0]
        bboxes = args[1]

        img_shape = image.shape

        bboxes = np.array(bboxes)

        bboxes[:, : 4] *= [self._resample_scale_x, self._resample_scale_y, self._resample_scale_x, self._resample_scale_y]
        # : 需要检查是否越界
        bboxes = clip_box(bboxes, image)
        self._aug_done()
        return bboxes.tolist(), 

class RandomResampleCombine(pAug.AugFunc):
    '''
     @brief RandomResampleImageAndBBoxes
    '''
    def __init__(self, scale=0.2, diff=False):
        self.scale = scale
        
        if type(self.scale) == tuple:
            assert len(self.scale) == 2, "Invalid range"
            assert self.scale[0] > -1, "Scale factor can't be less than -1"
            assert self.scale[1] > -1, "Scale factor can't be less than -1"
        else:
            assert self.scale > 0, "Please input a positive float"
            self.scale = (max(-1, -self.scale), self.scale)
        
        self.diff = diff

        self._image_scale = IR()
        self._bboxes_scale = Resample()
        pass

    def __call__(self, *args):
        img = args[0]
        bboxes = args[1]

        if self.diff:
            scale_x = random.uniform(*self.scale)
            scale_y = random.uniform(*self.scale)
        else:
            scale_x = random.uniform(*self.scale)
            scale_y = scale_x

        resize_scale_x = 1 + scale_x
        resize_scale_y = 1 + scale_y

        self._image_scale.resample_scale_x = resize_scale_x
        self._image_scale.resample_scale_y = resize_scale_y
        img_ret,  = self._image_scale(img)
        self._bboxes_scale.resample_scale_x = resize_scale_x
        self._bboxes_scale.resample_scale_y = resize_scale_y
        bboxes,  = self._bboxes_scale(img, bboxes)
        return img_ret, bboxes

    @property
    def doc(self):
        return 'resample the image and crop into the original image shape'

    @property
    def name(self):
        return 'Resample'


class Translate(IT):
    def __init__(self):
        IT.__init__(self)
 

    def __call__(self, *args):
        self.check_factor()
        image = args[0]
        bboxes = np.array(args[1])

        image_shape = image.shape
        
        translate_factor_x = self.translate_factor_x
        translate_factor_y = self.translate_factor_y
        
        canvas = np.zeros(image_shape).astype(np.uint8)

        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(translate_factor_x * image.shape[1])
        corner_y = int(translate_factor_y * image.shape[0])
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [
            max(corner_y, 0), 
            max(corner_x ,0), 
            min(image_shape[0], corner_y + image.shape[0]), 
            min(image_shape[1], corner_x + image.shape[1])
            ]

        mask = image[max(-corner_y, 0): min(image.shape[0], -corner_y + image_shape[0]), \
            max(-corner_x, 0): min(image.shape[1], -corner_x + image_shape[1]), :]
        canvas[orig_box_cords[0]: orig_box_cords[2], orig_box_cords[1]: orig_box_cords[3], :] = mask
        image = canvas
        
        bboxes[:, 0: 2] += [corner_x, corner_y]
        # : 需要检查是否越界
        bboxes = clip_box(bboxes, image)
        self._aug_done()
        return bboxes.tolist(),
    pass


class RandomTranslateConbine(object):
    def __init__(self, translate = 0.2, diff = False):
        '''
         @brief random translate which combine the ImageTranslate and the bboxesTranslate 
         @note
         @param[in] translate
         float, tuple with float
         while float, this means the translate range: [-translate, translate]
         while tuple, this means the translate range: [translate[0], translate[1]]
         @param[in] diff
         bool
         while False, x_translate and y_translate use the same random variable
         while True, the the translate and y_translate are from different random variable
        '''
        self.translate = translate
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1

        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)
            pass
        self.diff = diff

        self._image_translate = IT()
        self._bboxes_translate = Translate()

    def __call__(self, *args):
        '''
         @brief
         @note
         @param[in] args, with length 2
         [0]: image with shape [height, width[, channel]]
         [1]: bboxes, list with [x, y, width, height]
         @ret
        '''
        img = args[0]
        bboxes = args[1]

        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)
        
        if not self.diff:
            translate_factor_y = translate_factor_x
            pass

        self._image_translate.translate_factor_x = translate_factor_x
        self._image_translate.translate_factor_y = translate_factor_y
        img_ret,  = self._image_translate(img)

        self._bboxes_translate.translate_factor_x = translate_factor_x
        self._bboxes_translate.translate_factor_y = translate_factor_y
        bboxes,  = self._bboxes_translate(img, bboxes)
            
        return img_ret, bboxes

    @property
    def doc(self):
        return 'move the image and crop outof the field'

    @property
    def name(self):
        return 'Translate'

    
def get_corners(bboxes):
    width = bboxes[:, 2: 3]
    height = bboxes[:, 3: 4]
    x1 = bboxes[:, 0].reshape(-1, 1)
    y1 = bboxes[:, 1].reshape(-1, 1)
    x2 = x1 + width
    y2 = y1 
    x3 = x1
    y3 = y1 + height
    x4 = x1 + width
    y4 = y1 + height
    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    return corners


def rotate_box(corners, angle,  cx, cy, h, w):
    '''
     @brief Rotate the bounding box.
     @paran[in] corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
     @param[in] angle : float
        angle by which the image is to be rotated
     @param[in] cx : int
        x coordinate of the center of image (about which the box will be rotated)
     @param[in] cy : int
        y coordinate of the center of image (about which the box will be rotated)
     @param[in] h : int 
        height of the image
     @param[in] w : int 
        width of the image
     @ret numpy.ndarray
         Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
         corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    '''
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated


def get_enclosing_box(corners):
    '''
     @briefGet an enclosing box for ratated corners of a bounding box
     @param[in] corners : numpy.ndarray
         Numpy array of shape `N x 8` containing N bounding boxes each described by their 
         corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
     @ret numpy.ndarray
         Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
         number of bounding boxes and the bounding boxes are represented in the
         format `x1 y1 x2 y2 
    '''
    x_ = corners[:, [0, 2, 4, 6]]
    y_ = corners[:, [1, 3, 5, 7]]
    xmin = np.min(x_, 1).reshape(-1,1)
    ymin = np.min(y_, 1).reshape(-1,1)
    xmax = np.max(x_, 1).reshape(-1,1)
    ymax = np.max(y_, 1).reshape(-1,1)
    final = np.hstack((xmin, ymin, xmax, ymax, corners[:, 8: ]))
    return final


class Rotate(IRE):
    def __init__(self):
        IRE.__init__(self)
        pass

    def __call__(self, *args):
        '''
         @brief
         @note
         @param[in] args, *args
         [0]: image
         [1]: bboxes
        '''
        image = args[0]
        bboxes = np.array(args[1])

        angle = self.angle
        
        w, h = image.shape[1], image.shape[0]
        cx, cy = w // 2, h // 2
        
        corners = get_corners(bboxes)
        
        corners = np.hstack((corners, bboxes[:, 4: ]))

        corners[:, : 8] = rotate_box(corners[:, : 8], angle, cx, cy, h, w)
        
        new_bbox = get_enclosing_box(corners)
        
        img = rotate_im(image, angle)
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        
        new_bbox[:, : 4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y]
        new_bbox[:, 2: 4] = new_bbox[:, 2: 4] - new_bbox[:, 0: 2]
        bboxes  = new_bbox
        bboxes = clip_box(bboxes, image)
        self.aug_done()
        return bboxes.tolist(), 


class RandomRotateCombine(object):
    def __init__(self, angle = 10):
        self.angle = angle
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
        else:
            self.angle = (-self.angle, self.angle)
            pass
        self._image_rotate = IRE()
        self._bboxes_rotate = Rotate()
            
    def __call__(self, *args):
        image = args[0]
        bboxes = args[1]
    
        angle = random.uniform(*self.angle)
        
        self._image_rotate.angle = angle
        img,  = self._image_rotate(image)

        self._bboxes_rotate.angle = angle
        bboxes,  = self._bboxes_rotate(image, bboxes)
        return img, bboxes

    @property
    def doc(self):
        return 'rotate the image, and than resize to the original image shape'
    
    @property
    def name(self):
        return 'Rotate'
       

class Shear(IS):
    '''
     @note Shears an image in horizontal direction   
     Bounding boxes which have an area of less than 25% in the remaining in the 
     transformed image is dropped. The resolution is maintained, and the remaining
     area if any is filled by black color.
     @param[in] shear_factor: float
         Factor by which the image is sheared in the x-direction
     @ret
     numpy.ndarray
         Tranformed bounding box co-ordinates of the format `n x 4` where n is 
         number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    '''
    def __init__(self):
        IS.__init__(self)
        
    
    def __call__(self, *args):
        image = args[0]
        image_shape = image.shape
        bboxes = args[1]
        bboxes = np.array(bboxes)
        
        shear_factor = self.shear_factor_x

        if shear_factor < 0:
            bboxes, = HorizontalFlip()(image, bboxes)
            image, = IH()(image) 
            bboxes = np.array(bboxes)
            pass
        bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + (bboxes[:, [1, 3]] * abs(shear_factor))
        new_width = (image_shape[1] + image_shape[0] * shear_factor)
        fractor_width = image_shape[1] / new_width
        print(fractor_width)
        bboxes[:, [0, 2]] *= fractor_width
        if shear_factor < 0:
            bboxes, = HorizontalFlip()(image, bboxes.tolist())
            image, = IH()(image) 
            bboxes = np.array(bboxes)
            pass
        bboxes = clip_box(bboxes, image)
        self.aug_done()

        return bboxes.tolist(),
    

class CombineRandomShear(pAug.AugFunc):
    '''
     @note Randomly shears an image in horizontal direction   
     Bounding boxes which have an area of less than 25% in the remaining in the 
     transformed image is dropped. The resolution is maintained, and the remaining
     area if any is filled by black color.
     @param[in] shear_factor: float or tuple(float)
         if **float**, the image is sheared horizontally by a factor drawn 
         randomly from a range (-`shear_factor`, `shear_factor`). If **tuple**,
         the `shear_factor` is drawn randomly from values specified by the 
         tuple
     @ret 
     numpy.ndaaray
         Sheared image in the numpy format of shape `HxWxC`
     numpy.ndarray
         Tranformed bounding box co-ordinates of the format `n x 4` where n is 
         number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    '''
    def __init__(self, shear_factor = 0.2):
        pAug.AugFunc.__init__(self)
        self.shear_factor = shear_factor
        
        if type(self.shear_factor) == tuple:
            assert len(self.shear_factor) == 2, "Invalid range for scaling factor"   
        else:
            self.shear_factor = (-self.shear_factor, self.shear_factor)
            pass
        self._image_shear = IS()
        self._bboxes_shear = Shear()
        pass
        
        
    def __call__(self, *args):
        image = args[0]
        bboxes = args[1]
    
        shear_factor = random.uniform(*self.shear_factor)

        self._image_shear.shear_factor_x = shear_factor
        self._image_shear.shear_factor_y = shear_factor
        img, = self._image_shear(image)
        self._bboxes_shear.shear_factor_x = shear_factor
        self._bboxes_shear.shear_factor_y = shear_factor
        bboxes, = self._bboxes_shear(image, bboxes)
        return img, bboxes
    
    @property
    def doc(self):
        return 'shear the image follow the x and y'

    @property
    def name(self):
        return 'Shear'
        

class Resize(object):
    """Resize the image in accordance to `image_letter_box` function in darknet 
    
    The aspect ratio is maintained. The longer side is resized to the input 
    size of the network, while the remaining space on the shorter side is filled 
    with black color. **This should be the last transform**
    
    
    Parameters
    ----------
    inp_dim : tuple(int)
        tuple containing the size to which the image will be resized.
        
    Returns
    -------
    
    numpy.ndaaray
        Sheared image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """
    
    def __init__(self, inp_dim):
        self.inp_dim = inp_dim
        
    def __call__(self, img, bboxes):
        w,h = img.shape[1], img.shape[0]
        img = letterbox_image(img, self.inp_dim)
    
    
        scale = min(self.inp_dim/h, self.inp_dim/w)
        bboxes[:,:4] *= (scale)
    
        new_w = scale*w
        new_h = scale*h
        inp_dim = self.inp_dim   
    
        del_h = (inp_dim - new_h)/2
        del_w = (inp_dim - new_w)/2
    
        add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)
    
        bboxes[:,:4] += add_matrix
    
        img = img.astype(np.uint8)
    
        return img, bboxes 
    

class RandomHSV(object):
    """HSV Transform to vary hue saturation and brightness
    
    Hue has a range of 0-179
    Saturation and Brightness have a range of 0-255. 
    Chose the amount you want to change thhe above quantities accordingly. 
    
    
    
    
    Parameters
    ----------
    hue : None or int or tuple (int)
        If None, the hue of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-hue, hue) and added to the 
        hue of the image. If tuple, the int is sampled from the range 
        specified by the tuple.   
        
    saturation : None or int or tuple(int)
        If None, the saturation of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-saturation, saturation) 
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.   
        
    brightness : None or int or tuple(int)
        If None, the brightness of the image is left unchanged. If int, 
        a random int is uniformly sampled from (-brightness, brightness) 
        and added to the hue of the image. If tuple, the int is sampled
        from the range  specified by the tuple.   
    
    Returns
    -------
    
    numpy.ndaaray
        Transformed image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Resized bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """
    
    def __init__(self, hue = None, saturation = None, brightness = None):
        if hue:
            self.hue = hue 
        else:
            self.hue = 0
            
        if saturation:
            self.saturation = saturation 
        else:
            self.saturation = 0
            
        if brightness:
            self.brightness = brightness
        else:
            self.brightness = 0
            
            

        if type(self.hue) != tuple:
            self.hue = (-self.hue, self.hue)
            
        if type(self.saturation) != tuple:
            self.saturation = (-self.saturation, self.saturation)
        
        if type(brightness) != tuple:
            self.brightness = (-self.brightness, self.brightness)
    
    def __call__(self, img, bboxes):

        hue = random.randint(*self.hue)
        saturation = random.randint(*self.saturation)
        brightness = random.randint(*self.brightness)
        
        img = img.astype(int)
        
        a = np.array([hue, saturation, brightness]).astype(int)
        img += np.reshape(a, (1,1,3))
        
        img = np.clip(img, 0, 255)
        img[:,:,0] = np.clip(img[:,:,0],0, 179)
        
        img = img.astype(np.uint8)

        
        
        return img, bboxes
    
class Sequence(object):

    """Initialise Sequence object
    
    Apply a Sequence of transformations to the images/boxes.
    
    Parameters
    ----------
    augemnetations : list 
        List containing Transformation Objects in Sequence they are to be 
        applied
    
    probs : int or list 
        If **int**, the probability with which each of the transformation will 
        be applied. If **list**, the length must be equal to *augmentations*. 
        Each element of this list is the probability with which each 
        corresponding transformation is applied
    
    Returns
    -------
    
    Sequence
        Sequence Object 
        
    """
    def __init__(self, augmentations, probs = 1):

        
        self.augmentations = augmentations
        self.probs = probs
        
    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                images, bboxes = augmentation(images, bboxes)
        return images, bboxes