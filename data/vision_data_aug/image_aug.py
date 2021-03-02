# coding=utf-8
import copy
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


class ImageVerticalFlip:
    def __init__(self):
        pass


class VerticalFlip(ImageVerticalFlip, pAug.AugFunc):
    def __init__(self):
        ImageVerticalFlip.__init__(self)
        pAug.AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        img = args[0]
        img = img[::-1, :, :]
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
        
        canvas = np.zeros(img_shape, dtype = img.dtype)
        
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
    '''
     @brief base common class for Translate
     @note
     hanlde the translate_factor_x and the translate_factor_y
     translate_factor_x: the translate factor of width, sample in (-1, 1)
     translate_factor_y: the translate factor of height, sample in (-1, 1)
    '''
    def __init__(self, dtype=None):
        self._translate_factor_x = None
        self._translate_factor_y = None
        self._dtype = dtype
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
        
        canvas = np.zeros(img_shape).astype(img.dtype)

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

        #mask = img[max(-corner_y, 0): min(img.shape[0], -corner_y + img_shape[0]), \
        #    max(-corner_x, 0): min(img.shape[1], -corner_x + img_shape[1]), :]
        #canvas[orig_box_cords[0]: orig_box_cords[2], orig_box_cords[1]: orig_box_cords[3], :] = mask
        #img = canvas

        mask = img[max(-corner_y, 0): min(img.shape[0], -corner_y + img_shape[0]), \
            max(-corner_x, 0): min(img.shape[1], -corner_x + img_shape[1]), :]
        img = cv2.resize(mask, img.shape[0: 2], interpolation=cv2.INTER_CUBIC)
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


class ImageHSV:
    def __init__(self):
        pass

    def aug_done(self):
        self._hue = None
        self._saturation = None
        self._brightness = None
        pass

    def set_hue(self, hue):
        self._hue = hue
        pass

    def get_hue(self):
        return self._hue
    hue = property(get_hue, set_hue)

    def get_saturation(self):
        return self._saturation

    def set_saturation(self, saturation):
        self._saturation = saturation
    saturation = property(get_saturation, set_saturation)

    def get_brightness(self):
        return self._brightness

    def set_brightness(self, brightness):
        self._brightness = brightness
    brightness = property(get_brightness, set_brightness)


class HSV(ImageHSV, pAug.AugFunc):
    '''
     @note HSV Transform to vary hue saturation and brightness
     Hue has a range of 0-179
     Saturation and Brightness have a range of 0-255. 
     Chose the amount you want to change thhe above quantities accordingly. 
     @param[in] hue : None or int or tuple (int)
         If None, the hue of the image is left unchanged. If int, 
         a random int is uniformly sampled from (-hue, hue) and added to the 
         hue of the image. If tuple, the int is sampled from the range 
         specified by the tuple.   
     @param[in] saturation : None or int or tuple(int)
         If None, the saturation of the image is left unchanged. If int, 
         a random int is uniformly sampled from (-saturation, saturation) 
         and added to the hue of the image. If tuple, the int is sampled
         from the range  specified by the tuple.   
     @param[in] brightness : None or int or tuple(int)
         If None, the brightness of the image is left unchanged. If int, 
         a random int is uniformly sampled from (-brightness, brightness) 
         and added to the hue of the image. If tuple, the int is sampled
         from the range  specified by the tuple.   
     @ret
     numpy.ndaaray
         Transformed image in the numpy format of shape `HxWxC`
     
     numpy.ndarray
         Resized bounding box co-ordinates of the format `n x 4` where n is 
         number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
    '''
    def __init__(self, hue = None, saturation = None, brightness = None):
        ImageHSV.__init__(self)
        pAug.AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        image = args[0]

        hue = self.hue
        saturation = self.saturation
        brightness = self.brightness
        
        image = (image * 255).astype(np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int32)
        
        a = np.array([hue, saturation, brightness]).astype(np.int32)
        image += np.reshape(a, (1,1,3))
        
        image = np.clip(image, 0, 255)
        image[:, :, 0] = np.clip(image[:, :, 0], 0, 179)

        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return image, 
    pass


class ImageNoise:
    def __init__(self):
        self._mu = None
        self._sigma = None
        pass

    def set_sigma(self, sigma):
        self._sigma = sigma
        pass

    def get_sigma(self):
        return self._sigma
    sigma = property(get_sigma, set_sigma)

    def set_mu(self, mu):
        self._mu = mu
        pass

    def get_mu(self):
        return self._mu
    mu = property(get_mu, set_mu)
    pass


class Noise(ImageNoise, pAug.AugFunc):
    def __init__(self):
        ImageNoise.__init__(self)
        pAug.AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        image = args[0]
        shape = image.shape
        original_image_max_follow_channel = np.max(np.max(image, axis=0), axis=0)
        original_image_min_follow_channel = np.min(np.min(image, axis=0), axis=0)

        temp_image = np.random.normal(self._mu, self._sigma, shape) + image
        temp_image = np.clip(temp_image, 0, 255).astype(np.uint8)
        #image_max_follow_channel = np.max(np.max(temp_image, axis=0), axis=0)
        #image_min_follow_channel = np.min(np.min(temp_image, axis=0), axis=0)
        #temp_image = (temp_image - image_min_follow_channel) / (image_max_follow_channel - image_min_follow_channel)
        #temp_image = np.clip(temp_image, original_image_min_follow_channel / original_image_max_follow_channel, [1.0, 1.0, 1.0])
        #temp_image = temp_image * original_image_max_follow_channel
        #temp_image = temp_image.astype(image.dtype)

        #temp_image_2 = np.random.normal(self._mu, self._sigma, shape) + image
        #image_max_follow_channel = np.max(np.max(temp_image_2, axis=0), axis=0)
        #image_min_follow_channel = np.min(np.min(temp_image_2, axis=0), axis=0)
        #temp_image_2 = (temp_image_2 - image_min_follow_channel) / (image_max_follow_channel - image_min_follow_channel) * 255
        #temp_image_2 = temp_image_2.astype(image.dtype)
        return temp_image
    pass


class ImageSaltNoise:
    def __init__(self):
        self._prob = None
        self._thresh = None
        pass

    def get_prob(self):
        return self._prob

    def set_prob(self, prob):
        self._prob = prob
        self._thresh = 1.0 - self._prob
        pass
    prob = property(get_prob, set_prob)

    def get_thresh(self):
        return self._thresh
    thresh = property(get_thresh)
    pass


class SaltNoise(ImageSaltNoise, pAug.AugFunc):
    def __init__(self):
        ImageSaltNoise.__init__(self)
        pAug.AugFunc.__init__(self)
        pass

    def __call__(self, *args):
        image = copy.deepcopy(args[0])
        shift = np.random.random(image.shape)
        image[shift < self._prob] = 0
        image[shift > self._thresh] = 255
        return image


class ImageContrast:
    def __init__(self):
        self._contrast = None
        pass

    def set_contrast(self, contrast):
        self._contrast = contrast
    
    def get_contrast(self):
        return self._contrast
    contrast = property(get_contrast, set_contrast)

class Contrast(ImageContrast, pAug.AugFunc):
    def __init__(self):
        ImageContrast.__init__(self)
        pAug.AugFunc.__init__(self)

    def __call__(self, *args):
        image = args[0]
        return np.clip(image + (image - np.mean(image[:, :, 0] * 0.299) + np.mean(image[:, :, 1] * 0.587) + np.mean(image[:, :, 2] * 0.114)) * self._contrast / 255.0,  0, 255).astype(np.uint8)
