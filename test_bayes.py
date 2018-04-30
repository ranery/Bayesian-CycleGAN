# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.CycleGAN_bayes import CycleGAN
from util.visualizer import Visualizer
from util import html

opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

# load data
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

# load model
model = CycleGAN()
model.initialize(opt)
print('model [%s] was created.' % (model.name()))
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.inference()
    visuals = model.get_current_visuals()
    # loss = model.get_stye_loss()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    # visualizer.print_current_errors(0, 0, loss, 0)
    visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
