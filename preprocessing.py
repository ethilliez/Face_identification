import numpy as np
import math
from scipy import ndimage, misc
from skimage import transform, exposure
import re
import matplotlib.pyplot as plt
from glob import glob
from define_parameters import paths, image_parameters
import logging


class image_augmentation:
    def __init__(self):
        self.translation = image_parameters.TRANSLATION
        self.rotation = image_parameters.ROTATION
        self.shearing =  image_parameters.SHEARING
        self.contrast = image_parameters.CONTRAST

    def mirror_rotate(self, image, direction='horizontal'):
        if(direction == 'horizontal'):
            image_mirror = image[:,::-1,:]
        elif(direction == 'vertical'):
            image_mirror = image[::-1,:,:]
        return image_mirror

    def translate(self, image, shift):
        factor = int(shift*len(image))
        affine_tf = transform.AffineTransform(scale=(1.0,1.0),translation=(-factor,0))
        image_trans_left = transform.warp(image, inverse_map=affine_tf)
        affine_tf = transform.AffineTransform(scale=(1.0,1.0),translation=(0,-factor))
        image_trans_down = transform.warp(image, inverse_map=affine_tf)
        affine_tf = transform.AffineTransform(scale=(1.0,1.0),translation=(factor,0))
        image_trans_right = transform.warp(image, inverse_map=affine_tf)
        affine_tf = transform.AffineTransform(scale=(1.0,1.0),translation=(0,factor))
        image_trans_up = transform.warp(image, inverse_map=affine_tf)
        return image_trans_left, image_trans_right, image_trans_up, image_trans_down

    def rotate(self, image, angle):
        image_rotated = ndimage.rotate(image, angle, reshape=False)
       	return image_rotated

    def shear(self, image, factor, direction='horizontal'):
        affine_tf = transform.AffineTransform(shear=factor)
        # Apply transform to image data
        image_shear = transform.warp(image, inverse_map=affine_tf)    
        return image_shear

    def change_contrast(self, image, factor_gain, factor_bias):
        image_contrast = exposure.rescale_intensity(image, in_range=(factor_gain*factor_bias,(8+factor_gain)*factor_bias))
        return image_contrast

    def resize_image(self, image, npix):
        resized_image = misc.imresize(image,(npix,npix))
        return resized_image

    def perform_augmentation(self, image, npix):
        # Tranformation for each image
        # Resize original image
        image = self.resize_image(image, npix)
        # Perform mirror rotation and save
        image_m = self.mirror_rotate(image)
        # Perform translation on original image and save
        image_trans_left, image_trans_right, image_trans_up, image_trans_down = self.translate(image, self.translation)
        # Perform translation on mirror image and save
        image_trans_m_left, image_trans_m_right, image_trans_m_up, image_trans_m_down = self.translate(image_m, self.translation)
        # Perform rotation on original image and save
        image_rotated = self.rotate(image, self.rotation)
        # Perform rotation on mirror image and save
        image_m_rotated = self.rotate(image_m, self.rotation)
        # Perform shearing on original image and save
        image_shear = self.shear(image, self.shearing)
        # Perform shearing on mirror image and save
        image_m_shear = self.shear(image_m, self.shearing)
        # Perform change contrast on original image and save
        image_contrast = self.change_contrast(image, self.contrast[0], self.contrast[1])
        # Perform change contrast on mirror image and save
        image_m_contrast = self.change_contrast(image_m, self.contrast[0], self.contrast[1])
        # Save all images into a list
        images_augmented = [image_m,image_trans_left, image_trans_right, image_trans_up, image_trans_down,
               image_trans_m_left, image_trans_m_right, image_trans_m_up, image_trans_m_down,
               image_rotated, image_m_rotated, image_shear, image_m_shear, image_contrast, image_m_contrast]
        return images_augmented

