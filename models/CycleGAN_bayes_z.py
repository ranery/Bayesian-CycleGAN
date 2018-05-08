# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
# from torch.optim import lr_scheduler
import itertools
import util.util as util
from util.image_pool import ImagePool
from . import networks
import sys
import random
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


class CycleGAN():
    def name(self):
        return 'Bayesian CycleGAN Model'

    def initialize(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        if torch.cuda.is_available():
            print('cuda is available, we will use gpu!')
            self.Tensor = torch.cuda.FloatTensor
            torch.cuda.manual_seed_all(100)
        else:
            self.Tensor = torch.FloatTensor
            torch.manual_seed(100)
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        # get radio for network initialization
        ratio = 256 * 256 / opt.loadSize / (opt.loadSize / opt.ratio)

        # load network
        netG_input_nc = opt.input_nc + 1
        netG_output_nc = opt.output_nc + 1
        self.netG_A = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG_A,
                                        opt.n_downsample_global, opt.n_blocks_global, opt.norm).type(self.Tensor)
        self.netG_B = networks.define_G(netG_output_nc, opt.input_nc, opt.ngf, opt.netG_B,
                                        opt.n_downsample_global, opt.n_blocks_global, opt.norm).type(self.Tensor)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                            use_sigmoid, opt.num_D_A).type(self.Tensor)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                            use_sigmoid, opt.num_D_B).type(self.Tensor)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG_A, 'G_A', opt.which_epoch, self.save_dir)
            self.load_network(self.netG_B, 'G_B', opt.which_epoch, self.save_dir)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', opt.which_epoch, self.save_dir)
                self.load_network(self.netD_B, 'D_B', opt.which_epoch, self.save_dir)

        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss function
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('----------Network initialized!-----------')
        self.print_network(self.netG_A)
        self.print_network(self.netG_B)
        if self.isTrain:
            self.print_network(self.netD_A)
            self.print_network(self.netD_B)
        print('-----------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B']
        self.input_B = input['B' if AtoB else 'A']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A).type(self.Tensor)
        self.real_B = Variable(self.input_B).type(self.Tensor)

        # combine input image with random noise z
        self.real_B_zx = []
        for i in range(0, self.opt.mc_x):
            self.z_x = self.get_z_random(self.real_B[0, 1].size(), 'gauss')
            self.z_x = torch.unsqueeze(self.z_x, 0)
            self.z_x = torch.unsqueeze(self.z_x, 0)
            real_B_zx = []
            for i in range(0, self.opt.batchSize):
                _real = torch.unsqueeze(self.real_B[i], 0)
                _real = torch.cat([_real, self.z_x], dim=1)
                real_B_zx.append(_real)
            real_B_zx = torch.cat(real_B_zx)
            self.real_B_zx.append(real_B_zx)

        self.real_A_zy = []
        for i in range(0, self.opt.mc_y):
            self.z_y = self.get_z_random(self.real_A[0, 1].size(), 'gauss')
            self.z_y = torch.unsqueeze(self.z_y, 0)
            self.z_y = torch.unsqueeze(self.z_y, 0)
            real_A_zy = []
            for i in range(0, self.opt.batchSize):
                _real = torch.unsqueeze(self.real_A[i], 0)
                _real = torch.cat([_real, self.z_y], dim=1)
                real_A_zy.append(_real)
            real_A_zy = torch.cat(real_A_zy)
            self.real_A_zy.append(real_A_zy)

    def inference(self):
        real_A = Variable(self.input_A).type(self.Tensor)
        real_B = Variable(self.input_B).type(self.Tensor)

        # combine input image with random noise z
        real_B_zx = []
        z_x = self.get_z_random(real_B[0, 1].size(), 'gauss')
        z_x = torch.unsqueeze(z_x, 0)
        z_x = torch.unsqueeze(z_x, 0)
        for i in range(0, self.opt.batchSize):
            _real = torch.cat((real_B[i:i + 1], z_x), dim=1)
            real_B_zx.append(_real)
        real_B_zx = torch.cat(real_B_zx)

        real_A_zy = []
        z_y = self.get_z_random(real_A[0, 1].size(), 'gauss')
        z_y = torch.unsqueeze(z_y, 0)
        z_y = torch.unsqueeze(z_y, 0)
        for i in range(0, self.opt.batchSize):
            _real = torch.cat((real_A[i:i + 1], z_y), dim=1)
            real_A_zy.append(_real)
        real_A_zy = torch.cat(real_A_zy)

        # inference
        fake_B = self.netG_A(real_A_zy)
        fake_B_next = torch.cat((fake_B, z_x), dim=1)
        self.rec_A = self.netG_B(fake_B_next).data
        self.fake_B = fake_B.data

        fake_A = self.netG_B(real_B_zx)
        fake_A_next = torch.cat((fake_A, z_y), dim=1)
        self.rec_B = self.netG_A(fake_A_next).data
        self.fake_A = fake_A.data

    def get_image_paths(self):
        return self.image_paths

    def img_resize(self, img, target_width):
        ow, oh = img.size
        if (ow == target_width):
            return img
        else:
            w = target_width
            h = int(target_width * oh / ow)
        return img.resize((w, h), Image.BICUBIC)

    def get_z_random(self, nz, random_type='gauss'):
        z = self.Tensor(nz)
        if random_type == 'uni':
            z.copy_(torch.rand(nz) * 2.0 - 1.0)
        elif random_type == 'gauss':
            z.copy_(torch.randn(nz))
        z = Variable(z)
        return z

    def backward_G(self):
        # GAN loss D_A(G_A(A))
        fake_B = []
        for real_A in self.real_A_zy:
            _fake = self.netG_A(real_A)
            fake_B.append(_fake)
        fake_B = torch.cat(fake_B)

        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = []
        for real_B in self.real_B_zx:
            _fake = self.netG_B(real_B)
            fake_A.append(_fake)
        fake_A = torch.cat(fake_A)

        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # cycle loss
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # Forward cycle loss
        fake_B_next = []
        for i in range(0, self.opt.mc_y):
            _fake = fake_B[i * self.opt.batchSize:(i + 1) * self.opt.batchSize]
            _fake = torch.cat((_fake, self.z_x), dim=1)
            fake_B_next.append(_fake)
        fake_B_next = torch.cat(fake_B_next)

        rec_A = self.netG_B(fake_B_next)
        loss_cycle_A = 0
        for i in range(0, self.opt.mc_y):
            loss_cycle_A += self.criterionCycle(rec_A[i * self.opt.batchSize:(i + 1) * self.opt.batchSize],
                                                self.real_A) * lambda_A
        pred_cycle_G_A = self.netD_B(rec_A)
        loss_cycle_G_A = self.criterionGAN(pred_cycle_G_A, True)

        # Backward cycle loss
        fake_A_next = []
        for i in range(0, self.opt.mc_x):
            _fake = fake_A[i * self.opt.batchSize:(i + 1) * self.opt.batchSize]
            _fake = torch.cat((_fake, self.z_y), dim=1)
            fake_A_next.append(_fake)
        fake_A_next = torch.cat(fake_A_next)

        rec_B = self.netG_A(fake_A_next)
        loss_cycle_B = 0
        for i in range(0, self.opt.mc_x):
            loss_cycle_B += self.criterionCycle(rec_B[i * self.opt.batchSize:(i + 1) * self.opt.batchSize],
                                                self.real_B) * lambda_B
        pred_cycle_G_B = self.netD_A(rec_B)
        loss_cycle_G_B = self.criterionGAN(pred_cycle_G_B, True)

        # prior loss
        prior_loss_G_A = self.get_prior(self.netG_A.parameters(), self.opt.batchSize)
        prior_loss_G_B = self.get_prior(self.netG_B.parameters(), self.opt.batchSize)

        # total loss
        loss_G = loss_G_A + loss_G_B + (prior_loss_G_A + prior_loss_G_B) + (loss_cycle_G_A + loss_cycle_G_B) * self.opt.gamma + (loss_cycle_A + loss_cycle_B)
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.data[0] + loss_cycle_G_A.data[0] * self.opt.gamma + prior_loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0] + loss_cycle_G_B.data[0] * self.opt.gamma + prior_loss_G_A.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]

    def backward_D_A(self):
        fake_B = Variable(self.fake_B).type(self.Tensor)
        rec_B = Variable(self.rec_B).type(self.Tensor)
        # how well it classifiers fake images
        pred_fake = self.netD_A(fake_B.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_cycle_fake = self.netD_A(rec_B.detach())
        loss_D_cycle_fake = self.criterionGAN(pred_cycle_fake, False)

        # how well it classifiers real images
        pred_real = self.netD_A(self.real_B)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.mc_y

        # prior loss
        prior_loss_D_A = self.get_prior(self.netD_A.parameters(), self.opt.batchSize)

        # total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5 + (loss_D_real + loss_D_cycle_fake) * 0.5 * self.opt.gamma + prior_loss_D_A
        loss_D_A.backward()
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = Variable(self.fake_A).type(self.Tensor)
        rec_A = Variable(self.rec_A).type(self.Tensor)
        # how well it classifiers fake images
        pred_fake = self.netD_B(fake_A.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        pred_cycle_fake = self.netD_B(rec_A.detach())
        loss_D_cycle_fake = self.criterionGAN(pred_cycle_fake, False)

        # how well it classifiers real images
        pred_real = self.netD_B(self.real_A)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.mc_x

        # prior loss
        prior_loss_D_B = self.get_prior(self.netD_B.parameters(), self.opt.batchSize)

        # total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5 + (loss_D_real + loss_D_cycle_fake) * 0.5 * self.opt.gamma + prior_loss_D_B
        loss_D_B.backward()
        self.loss_D_B = loss_D_B.data[0]

    def backward_G_pair(self):
        # GAN loss D_A(G_A(A)) and L1 loss
        fake_B = []
        loss_G_A_L1 = 0
        for real_A in self.real_A_zy:
            _fake = self.netG_A(real_A)
            loss_G_A_L1 += self.criterionL1(_fake, self.real_B) * self.opt.lambda_A
            fake_B.append(_fake)
        fake_B = torch.cat(fake_B)

        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        # GAN loss D_B(G_B(B))
        fake_A = []
        loss_G_B_L1 = 0
        for real_B in self.real_B_zx:
            _fake = self.netG_B(real_B)
            loss_G_B_L1 += self.criterionL1(_fake, self.real_A) * self.opt.lambda_B
            fake_A.append(_fake)
        fake_A = torch.cat(fake_A)

        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        # prior loss
        prior_loss_G_A = self.get_prior(self.netG_A.parameters(), self.opt.batchSize)
        prior_loss_G_B = self.get_prior(self.netG_B.parameters(), self.opt.batchSize)

        # total loss
        loss_G = loss_G_A + loss_G_B + (prior_loss_G_A + prior_loss_G_B) + (loss_G_A_L1 + loss_G_B_L1)
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data

    def backward_D_A_pair(self):
        fake_B = Variable(self.fake_B).type(self.Tensor)
        # how well it classifiers fake images
        pred_fake = self.netD_A(fake_B.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # how well it classifiers real images
        pred_real = self.netD_A(self.real_B)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.mc_y

        # prior loss
        prior_loss_D_A = self.get_prior(self.netD_A.parameters(), self.opt.batchSize)

        # total loss
        loss_D_A = (loss_D_real + loss_D_fake) * 0.5 + prior_loss_D_A
        loss_D_A.backward()

    def backward_D_B_pair(self):
        fake_A = Variable(self.fake_A).type(self.Tensor)
        # how well it classifiers fake images
        pred_fake = self.netD_B(fake_A.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # how well it classifiers real images
        pred_real = self.netD_B(self.real_A)
        loss_D_real = self.criterionGAN(pred_real, True) * self.opt.mc_x

        # prior loss
        prior_loss_D_B = self.get_prior(self.netD_B.parameters(), self.opt.batchSize)

        # total loss
        loss_D_B = (loss_D_real + loss_D_fake) * 0.5 + prior_loss_D_B
        loss_D_B.backward()

    def optimize(self, pair=False):
        # forward
        self.forward()
        # G_A and G_B
        # E_A and E_B
        self.optimizer_G.zero_grad()
        if pair == True:
            self.backward_G_pair()
        else:
            self.backward_G()
        self.optimizer_G.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        if pair == True:
            self.backward_D_A_pair()
        else:
            self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        if pair == True:
            self.backward_D_B_pair()
        else:
            self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_loss(self):
        loss = OrderedDict([
            ('D_A', self.loss_D_A),
            ('D_B', self.loss_D_B),
            ('G_A', self.loss_G_A),
            ('G_B', self.loss_G_B)
        ])
        if self.opt.gamma == 0:
            loss['cyc_A'] = self.loss_cycle_A
            loss['cyc_B'] = self.loss_cycle_B
        elif self.opt.gamma > 0:
            loss['cyc_G_A'] = self.loss_cycle_A
            loss['cyc_G_B'] = self.loss_cycle_B
        return loss

    def get_stye_loss(self):
        loss = OrderedDict([
            ('L1_A', self.loss_G_A_L1),
            ('L1_B', self.loss_G_B_L1)
        ])
        return loss

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        visuals = OrderedDict([
            ('real_A', real_A),
            ('fake_B', fake_B),
            ('rec_A', rec_A),
            ('real_B', real_B),
            ('fake_A', fake_A),
            ('rec_B', rec_B)
        ])
        return visuals

    def get_prior(self, parameters, dataset_size):
        prior_loss = Variable(torch.zeros((1))).cuda()
        for param in parameters:
            prior_loss += torch.mean(param * param)
        return prior_loss / dataset_size

    # def get_noise(self, parameters, alpha, dataset_size):
    #     noise_loss = Variable(torch.zeros((1))).cuda()
    #     noise_std = np.sqrt(2 * alpha)
    #     for param in parameters:
    #         noise = Variable(torch.normal(std=torch.ones(param.size()))).cuda()
    #         noise_loss += torch.sum(param*noise*noise_std)
    #     return noise_loss / dataset_size

    def save_model(self, label):
        self.save_network(self.netG_A, 'G_A', label)
        self.save_network(self.netG_B, 'G_B', label)
        self.save_network(self.netD_A, 'D_A', label)
        self.save_network(self.netD_B, 'D_B', label)

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        try:
            network.load_state_dict(torch.load(save_path))
        except:
            pretrained_dict = torch.load(save_path)
            model_dict = network.state_dict()
            try:
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                network.load_state_dict(pretrained_dict)
                print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
            except:
                print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                if sys.version_info >= (3, 0):
                    not_initialized = set()
                else:
                    from sets import Set
                    not_initialized = Set()
                for k, v in pretrained_dict.items():
                    if v.size() == model_dict[k].size():
                        model_dict[k] = v

                for k, v in model_dict.items():
                    if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                        not_initialized.add(k.split('.')[0])
                print(sorted(not_initialized))
                network.load_state_dict(model_dict)

    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if torch.cuda.is_available():
            network.cuda()

    def print_network(self, net):
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)

    # update learning rate (called once every iter)
    def update_learning_rate(self, epoch, epoch_iter, dataset_size):
        # lrd = self.opt.lr / self.opt.niter_decay
        if epoch > self.opt.niter:
            lr = self.opt.lr * np.exp(-1.0 * min(1.0, epoch_iter / float(dataset_size)))
            for param_group in self.optimizer_D_A.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_D_B.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_G.param_groups:
                param_group['lr'] = lr
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
            self.old_lr = lr
        else:
            lr = self.old_lr

