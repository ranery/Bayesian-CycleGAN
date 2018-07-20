# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 17:34:25 2017

@author: dell-pc
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import  Image
import cv2
import os

color2index = {
        (0,     0,   0) : 0,
        (0,     0,   0) : 1,
        (0,     0,   0) : 2,
        (0,     0,   0) : 3,
        (0,     0,   0) : 4,
        (111,  74,   0) : 5,
        (81,    0,  81) : 6,
        (128,  64, 128) : 7,
        (244,  35, 232) : 8,
        (250, 170, 160) : 9,
        (230, 150, 140) : 10,
        (70 ,  70,  70) : 11,
        (102, 102, 156) : 12,
        (190, 153, 153) : 13, 
        (180, 165, 180) : 14,
        (150, 100, 100) : 15,
        (150, 120,  90) : 16,
        (153, 153, 153) : 17,
        (153, 153, 153) : 18,
        (250, 170,  30) : 19,
        (220, 220,   0) : 20,
        (107, 142,  35) : 21,
        (152, 251, 152) : 22,
        ( 70, 130, 180) : 23,
        (220,  20,  60) : 24,
        (255,   0,   0) : 25,
        (  0,   0, 142) : 26,
        (  0,   0,  70) : 27,
        (  0,  60, 100) : 28,
        (  0,   0,  90) : 29,
        (  0,   0, 110) : 30,
        (  0,  80, 100) : 31,
        (  0,   0, 230) : 32,
        (119,  11,  32) : 33
        #(  0,  0,  142) : -1
}
    
def im2index(im):
    """
    turn a 3 channel RGB image to 1 channel index image
    """
    assert len(im.shape) == 3
    height, width, ch = im.shape
    assert ch == 3
    m_lable = np.zeros((height, width, 1), dtype=np.uint8)
    for w in range(width):
        for h in range(height):
            r, g, b = im[h, w, :]
            m_lable[h, w, 0] = color2index[(r, g, b)]
    return m_lable
    
def img_extract(img_file, path_save):
    os.chdir(path)
    img = Image.open(img_file)
    img = np.array(img, dtype=np.float32)
    img = img[:, :, 0:3]
    label = im2index(img)
    label = np.array(label, dtype=np.uint8)
    os.chdir(path_save)
    cv2.imwrite(img_file, label)

# color = Image.open('/root/nfs-datasets/cityscapes/test/color.png')
# color = np.array(color, dtype=np.float32)
# color = color[:, :, 0:3]
#
# label = im2index(color)
# label = np.array(label, dtype=np.uint8)
# cv2.imwrite('label.png', label)

path = '/root/nfs-datasets/cityscapes/trainB/'
path_save = '/root/nfs-datasets/cityscapes/trainB_label/'
if not os.path.exists(path_save):
    os.mkdir(path_save)

num_file = 0
for filename in os.listdir(path):
    num_file += 1
    img_extract(filename, path_save)