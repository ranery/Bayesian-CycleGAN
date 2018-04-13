# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.CycleGAN_bayes import CycleGAN
from util.visualizer import Visualizer


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
# iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
if opt.continue_train:
    # try:
    #     start_epoch, epoch_iter = np.loadtxt(iter_path , delimiter=',', dtype=int)
    # except:
    #     start_epoch, epoch_iter = 1, 0
    start_epoch = 14
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

# train
total_steps = (start_epoch-1) * dataset_size + epoch_iter
for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        model.set_input(data)
        model.optimize()

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

