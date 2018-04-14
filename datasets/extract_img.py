# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import os
import shutil

nof = open('female.txt')
yesf = open('male.txt')

noline = nof.readline()
yesline = yesf.readline()

list = os.listdir('./img_align_celeba/')
list.sort()
yesgo = True
nogo = True
for i in range(0, len(list)):
    imgName = os.path.basename(list[i])
    if os.path.splitext(imgName)[1] != '.jpg':
        continue
    noarray = noline.split()
    if len(noarray) < 1:
        nogo = False
    yesarray = yesline.split()
    if len(yesarray) < 1:
        yesgo = False
    if nogo and (imgName == noarray[0]):
        oldname = './img_align_celeba/' + imgName
        newname = './female/' + imgName
        shutil.move(oldname, newname)
        noline = nof.readline()
    elif yesgo and (imgName == yesarray[0]):
        oldname = './img_align_celeba/' + imgName
        newname = './male/' + imgName
        shutil.move(oldname, newname)
        yesline = yesf.readline()

    if i % 100 == 0:
        print(imgName)

nof.close()
yesf.close()