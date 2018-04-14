#!/usr/bin/env python3

import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import fid
from scipy.misc import imread
import tensorflow as tf
from inception_score import get_inception_score

########
# PATHS
########
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, help='data dir')
opt = parser.parse_args()
data_path = opt.dir
# '/home/zhoupan/disk1/cityscapesHD/testB_resize/' # set path to training set images
# output_path = 'inception_stats.npz' # path for where to store the statistics
# if you have downloaded and extracted
#   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
# set this path to the directory where the extracted files are, otherwise
# just set it to None and the script will later download the files for you
# inception_path = None
# print("check for inception model..", end=" ", flush=True)
# inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
# print("ok")

# loads all images into memory (this might require a lot of RAM!)
print("load images..", end=" " , flush=True)
image_list = glob.glob(os.path.join(data_path, '*.png'))
images = [imread(str(fn)).astype(np.float32) for fn in image_list]
print("%d images found and loaded" % len(images))

# print("create inception graph..", end=" ", flush=True)
# fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
# print("ok")

print("calculte Inception stats..", end=" ", flush=True)
mu, sigma = get_inception_score(images) # fid.calculate_activation_statistics(images, sess, batch_size=1)
print('mu: ', mu)
print('sigam: ', sigma)
# np.savez_compressed(output_path, mu=mu, sigma=sigma)
print("finished")
