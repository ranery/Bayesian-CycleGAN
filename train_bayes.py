# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import time
import os
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.CycleGAN_bayes import CycleGAN
from util.visualizer import Visualizer
import random
import torch
from PIL import Image
import torchvision.transforms as transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load data
opt = TrainOptions().parse()
# opt.serial_batches = True
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# load model
model = CycleGAN()
model.initialize(opt)
print('model [%s] was created.' % (model.name()))
visualizer = Visualizer(opt)

# continue train or not
if opt.continue_train:
    start_epoch = 15
    epoch_iter = 0
    print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
else:
    start_epoch, epoch_iter = 1, 0

# use for debug
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10

# pre-train for cityscapes if use semi-supervised
if opt.need_match:
    # paired data path and name list
    origin_path = os.getcwd()
    path_A = opt.dataroot + '/pairA'
    path_B = opt.dataroot + '/pairB'
    list_A = os.listdir(path_A)
    list_B = os.listdir(path_B)

    # for paired data
    for i in range(0, len(list_A)):
        os.chdir(path_A)
        A = Image.open(list_A[i]).convert('RGB')
        os.chdir(path_B)
        B = Image.open(list_B[i]).convert('RGB')
        A = transform(model.img_resize(A, opt.loadSize))
        B = transform(model.img_resize(B, opt.loadSize))
        A = torch.unsqueeze(A, 0)
        B = torch.unsqueeze(B, 0)
        data = {'A': A, 'B': B, 'A_paths': path_A, 'B_paths': path_B}

        os.chdir(origin_path)
        model.set_input(data)
        model.optimize(pair=True)

# train
total_steps = (start_epoch-1) * dataset_size + epoch_iter
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    # for unpaired data
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        model.set_input(data)
        model.optimize(pair=False)

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        if total_steps % opt.print_freq == 0:
            loss = model.get_current_loss()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, loss, t)
            model.update_learning_rate(epoch, epoch_iter, dataset_size)
            if opt.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, loss)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save_model('latest')


        if total_steps % 10000 == 0:
            print('saving model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save_model(str(total_steps))

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

