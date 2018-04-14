# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import os
import os.path
import random
from PIL import Image

list_male = os.listdir('./male/')
list_female = os.listdir('./female/')

train_male = random.sample(list_male, 2000)
train_female = random.sample(list_female, 2000)

list_male = [i for i in list_male if i not in train_male]
list_female = [i for i in list_female if i not in train_female]

test_male = random.sample(list_male, 500)
test_female = random.sample(list_female, 500)

def img_extract(img_file, path, path_save):
    origin_path = os.getcwd()
    os.chdir(path)
    img = Image.open(img_file)
    img = img_resize(img, 256)
    os.chdir(origin_path)
    img.save(os.path.join(path_save, os.path.basename(img_file)))

def img_resize(img, target_width):
    return img.resize((target_width, target_width), Image.BICUBIC)

male_path = './male/'
female_path = './female/'
trainA_path = './male2female/trainA/'
trainB_path = './male2female/trainB/'
testA_path = './male2female/testA/'
testB_path = './male2female/testB/'

for male in train_male:
    img_extract(male, male_path, trainA_path)

for female in train_female:
    img_extract(female, female_path, trainB_path)

for male in test_male:
    img_extract(male, male_path, testA_path)

for female in test_female:
    img_extract(female, female_path, testB_path)

