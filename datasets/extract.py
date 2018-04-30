# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import os
import os.path
import argparse
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, default='fake_B', help='generate file type')
parser.add_argument('--dir', type=str)
parser.add_argument('--save_dir', type=str)

opt = parser.parse_args()

def img_extract(img_file, path_save):
    os.chdir(opt.dir)    
    if opt.type in img_file:
        img = Image.open(img_file)
        img.save(os.path.join(path_save, os.path.basename(img_file)))


path_save = opt.save_dir  + opt.type + '/'
if not os.path.exists(path_save):
	os.mkdir(path_save)

num_file = 0
for filename in os.listdir(opt.dir):
    num_file += 1
    img_extract(filename, path_save)
