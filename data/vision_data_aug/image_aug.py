# coding=utf-8
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import Putil.data.aug as pAug


class ImageHorizontalFlip:
    def __init__(self):
        pass


class HorizontalFlip(ImageHorizontalFlip, pAug.AugFunc):
    def __init__(self):
        ImageHorizontalFlip.__init__(self)
        pAug.AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        img = args[0]
        img = img[:, ::-1, :]
        return img, 


class ImageResample:
    def __init__(self):
        self._resample_scale_x = None
        self._resample_scale_y = None
        pass

    def set_resample_scale_x(self, resample_x):
        self._resample_scale_x = resample_x
        pass

    def get_resample_scale_x(self):
        return self._resample_scale_x
    resample_scale_x = property(get_resample_scale_x, set_resample_scale_x)

    def set_resample_scale_y(self, resample_scale_y):
        self._resample_scale_y = resample_scale_y
        pass

    def get_resample_scale_y(self):
        return self._resample_scale_y
    resample_scale_y = property(get_resample_scale_y, set_resample_scale_y)

    def _aug_done(self):
        self._resample_scale_x = None
        self._resample_scale_y = None
        pass


class Resample(ImageResample, pAug.AugFunc):
    '''
     @brief
     @note 
     this function resize the image with [resize_scale_x, resize_scale_y]
     @param[in] resize_scale_x
     the resize_scale follow width
     @param[in] resize_scale_y
     the resize_scale follow height
     @param[in] img
     the target img with shape [height, width[, channel]], we only need the height and width
    '''
    def __init__(self, resample_method=None):
        ImageResample.__init__(self)
        pAug.AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        img = args[0]

        img_shape = img.shape

        img=  cv2.resize(img, None, fx = self._resample_scale_x, fy = self._resample_scale_y)
        
        canvas = np.zeros(img_shape, dtype = np.uint8)
        
        y_lim = int(min(self._resample_scale_y, 1) * img_shape[0])
        x_lim = int(min(self._resample_scale_x, 1) * img_shape[1])
        
        
        canvas[: y_lim, : x_lim, :] =  img[: y_lim, : x_lim, :]
        
        img = canvas
        self._aug_done()
        return img, 


class RandomTranslate(pAug.AugFunc):
    def __init__(self, translate = 0.2, diff = False):
        self.translate = translate
        
        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"  
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)
        self.diff = diff

    def __call__(self, img, bboxes):        
        #Chose a random digit to scale by 
        img_shape = img.shape
        
        #translate the image
        
        #percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)
        
        if not self.diff:
            translate_factor_y = translate_factor_x
            
        canvas = np.zeros(img_shape).astype(np.uint8)
    
        corner_x = int(translate_factor_x*img.shape[1])
        corner_y = int(translate_factor_y*img.shape[0])
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [max(0,corner_y), max(corner_x,0), min(img_shape[0], corner_y + img.shape[0]), min(img_shape[1],corner_x + img.shape[1])]
    
        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
        img = canvas
        
        bboxes[:,:4] += [corner_x, corner_y, corner_x, corner_y]
        
        bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]], 0.25)
        return img, bboxes


class ImageTranslate:
    def __init__(self):
        self._translate_factor_x = None
        self._translate_factor_y = None
        pass

    def set_translate_factor_x(self, translate_x):
        self._translate_factor_x = translate_x
        pass

    def get_translate_factor_x(self):
        return self._translate_factor_x
    translate_factor_x = property(get_translate_factor_x, set_translate_factor_x)

    def set_translate_factor_y(self, translate_y):
        self._translate_factor_y = translate_y
        pass

    def get_translate_factor_y(self):
        return self._translate_factor_y
    translate_factor_y = property(get_translate_factor_y, set_translate_factor_y)

    def _aug_done(self):
        self._translate_factor_x = None
        self._translate_factor_y = None

    def check_factor(self):
        assert self.translate_factor_x > -1 and self.translate_factor_x < 1, print('traslate_factor_x: {0}'.format(self._translate_factor_x))
        assert self.translate_factor_y > -1 and self.translate_factor_y < 1, print('traslate_factor_y: {0}'.format(self._translate_factor_y))
    pass
    

class Translate(ImageTranslate, pAug.AugFunc):
    def __init__(self):
        ImageTranslate.__init__(self)
        pAug.AugFunc.__init__(self)

    def __call__(self, *args):
        self.check_factor()

        img = args[0]

        img_shape = img.shape
        
        #translate the image
        
        #percentage of the dimension of the image to translate
        translate_factor_x = self.translate_factor_x
        translate_factor_y = self.translate_factor_y
        
        canvas = np.zeros(img_shape).astype(np.uint8)

        #get the top-left corner co-ordinates of the shifted box 
        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])
        
        #change the origin to the top-left corner of the translated box
        orig_box_cords =  [
            max(corner_y, 0), 
            max(corner_x ,0), 
            min(img_shape[0], corner_y + img.shape[0]), 
            min(img_shape[1], corner_x + img.shape[1])
            ]

        mask = img[max(-corner_y, 0): min(img.shape[0], -corner_y + img_shape[0]), \
            max(-corner_x, 0): min(img.shape[1], -corner_x + img_shape[1]), :]
        canvas[orig_box_cords[0]: orig_box_cords[2], orig_box_cords[1]: orig_box_cords[3], :] = mask
        img = canvas
        
        return img,
    
    
class RandomRotate(object):
    def __init__(self, angle = 10):
        self.angle = angle
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
            
        else:
            self.angle = (-self.angle, self.angle)
            
    def __call__(self, img, bboxes):
    
        angle = random.uniform(*self.angle)
    
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
    
        img = rotate_im(img, angle)
    
        corners = get_corners(bboxes)
    
        corners = np.hstack((corners, bboxes[:,4:]))
    
    
        corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)
    
        new_bbox = get_enclosing_box(corners)
    
    
        scale_factor_x = img.shape[1] / w
    
        scale_factor_y = img.shape[0] / h
    
        img = cv2.resize(img, (w,h))
    
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
    
        bboxes  = new_bbox
    
        bboxes = clip_box(bboxes, [0,0,w, h], 0.25)
    
        return img, bboxes

    

class ImageRotate:
    def __init__(self):
        self._angle = None
        pass

    def get_angle(self):
        return self._angle
    
    def set_angle(self, angle):
        self._angle = angle
        pass
    angle = property(get_angle, set_angle)

    def aug_done(self):
        self._angle = None
        pass
    pass


def rotate_im(image, angle):
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    return image


class Rotate(ImageRotate, pAug.AugFunc):
    def __init__(self):
        ImageRotate.__init__(self)
        pAug.AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        """
         @brief
         @note
         @param[in] args *args
         [0] img, the image with shape [height, width[, channel]]
        """
        img = args[0]

        angle = self.angle
        
        w, h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
        
        img = rotate_im(img, angle)
        scale_factor_x = img.shape[1] / w
        scale_factor_y = img.shape[0] / h
        
        img = cv2.resize(img, (w,h))

        self.aug_done()
        return img,
        

class ImageShear:
    def __init__(self):
        self.aug_done()
        pass

    def get_shear_factor_x(self):
        return self._shear_factor_x

    def set_shear_factor_x(self, shear_factor_x):
        self._shear_factor_x = shear_factor_x
    shear_factor_x = property(get_shear_factor_x, set_shear_factor_x)

    def get_shear_factor_y(self):
        return self._shear_factor_y
    
    def set_shear_factor_y(self, shear_factor_y):
        self._shear_factor_y = shear_factor_y
        pass
    shear_factor_y = property(get_shear_factor_y, set_shear_factor_y)

    def aug_done(self):
        self._shear_factor_x = None 
        self._shear_factor_y = None
        pass

        
class Shear(ImageShear, pAug.AugFunc):
    '''
     @brief Shears an image in horizontal direction   
     Bounding boxes which have an area of less than 25% in the remaining in the 
     transformed image is dropped. The resolution is maintained, and the remaining
     area if any is filled by black color.
     @param[in] shear_factor: float
         Factor by which the image is sheared in the x-direction
     @ret
     numpy.ndaaray
         Sheared image in the numpy format of shape `HxWxC`
    '''
    def __init__(self):
        ImageShear.__init__(self)
        pAug.AugFunc.__init__(self)
        pass
    
    def __call__(self, *args):
        image = args[0]
        image_shape = image.shape

        shear_factor = self.shear_factor_x
        if shear_factor < 0:
            image, = HorizontalFlip()(image)

        
        M = np.array([[1, abs(shear_factor), 0], [0, 1, 0]])
                
        nW =  image.shape[1] + abs(shear_factor * image.shape[0])
        
        image = cv2.warpAffine(image, M, (int(nW), image.shape[0]))
        
        if shear_factor < 0:
             image, = HorizontalFlip()(image)
            
        image = cv2.resize(image, image_shape[0: 2])

        self.aug_done() 
        return image,
    
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