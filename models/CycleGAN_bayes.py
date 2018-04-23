# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
from torch.optim import lr_scheduler
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
        netG_input_nc = opt.input_nc + 8
        netG_output_nc = opt.output_nc + 8
        self.netG_A = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG_A,
                                        opt.n_downsample_global, opt.n_blocks_global, opt.norm).type(self.Tensor)
        self.netG_B = networks.define_G(netG_output_nc, opt.input_nc, opt.ngf, opt.netG_B,
                                        opt.n_downsample_global, opt.n_blocks_global, opt.norm).type(self.Tensor)

        self.netE_A = networks.define_G(opt.input_nc, 8, 64, 'encoder', norm=opt.norm, ratio=ratio).type(self.Tensor)
        self.netE_B = networks.define_G(opt.output_nc, 8, 64, 'encoder', norm=opt.norm, ratio=ratio).type(self.Tensor)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                            use_sigmoid, opt.num_D_A).type(self.Tensor)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.n_layers_D, opt.norm,
                                            use_sigmoid, opt.num_D_B).type(self.Tensor)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG_A, 'G_A', opt.which_epoch, self.save_dir)
            self.load_network(self.netG_B, 'G_B', opt.which_epoch, self.save_dir)
            self.load_network(self.netE_A, 'E_A', opt.which_epoch, self.save_dir)
            self.load_network(self.netE_B, 'E_B', opt.which_epoch, self.save_dir)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', opt.which_epoch, self.save_dir)
                self.load_network(self.netD_B, 'D_B', opt.which_epoch, self.save_dir)

        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss function
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_A = torch.optim.Adam(self.netE_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_E_B = torch.optim.Adam(self.netE_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        print('----------Network initialized!-----------')
        self.print_network(self.netG_A)
        self.print_network(self.netG_B)
        self.print_network(self.netE_A)
        self.print_network(self.netE_B)
        if self.isTrain:
            self.print_network(self.netD_A)
            self.print_network(self.netD_B)
        print('-----------------------------------------')

        # dataset path and name list
        self.origin_path = os.getcwd()
        self.path_A = self.opt.dataroot + '/trainA'
        self.path_B = self.opt.dataroot + '/trainB'
        self.list_A = os.listdir(self.path_A)
        self.list_B = os.listdir(self.path_B)

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        self.input_A = input['A' if AtoB else 'B']
        self.input_B = input['B' if AtoB else 'A']
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A).type(self.Tensor)
        self.real_B = Variable(self.input_B).type(self.Tensor)

        # feature map
        mc_sample_x = random.sample(self.list_A, self.opt.mc_x)
        mc_sample_y = random.sample(self.list_B, self.opt.mc_y)
        self.real_B_zx = []
        self.real_A_zy = []
        os.chdir(self.path_A)
        for sample_x in mc_sample_x:
            z_x = Image.open(sample_x).convert('RGB')
            z_x = self.img_resize(z_x, self.opt.loadSize)
            z_x = transform(z_x)
            if self.opt.input_nc == 1:  # RGB to gray
                z_x = z_x[0, ...] * 0.299 + z_x[1, ...] * 0.587 + z_x[2, ...] * 0.114
                z_x = z_x.unsqueeze(0)
            z_x = Variable(z_x).type(self.Tensor)
            z_x = torch.unsqueeze(z_x, 0)

            self.mu_x, self.logvar_x = self.netE_A.forward(z_x)
            std = self.logvar_x.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
            latent_code = eps.mul(std).add_(self.mu_x)
            feat_map = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
                latent_code.size(0), latent_code.size(1), z_x.size(2), z_x.size(3))

            self.feat_map_zx = feat_map
            real_B_zx = []
            for i in range(0, self.opt.batchSize):
                _real = torch.unsqueeze(self.real_B[i], 0)
                _real = torch.cat([_real, feat_map], dim=1)
                real_B_zx.append(_real)
            real_B_zx = torch.cat(real_B_zx)
            self.real_B_zx.append(real_B_zx)

        os.chdir(self.path_B)
        self.feat_map_zy = []
        for sample_y in mc_sample_y:
            z_y = Image.open(sample_y).convert('RGB')
            z_y = self.img_resize(z_y, self.opt.loadSize)
            z_y = transform(z_y)
            if self.opt.output_nc == 1:  # RGB to gray
                z_y = z_y[0, ...] * 0.299 + z_y[1, ...] * 0.587 + z_y[2, ...] * 0.114
                z_y = z_y.unsqueeze(0)
            z_y = Variable(z_y).type(self.Tensor)
            z_y = torch.unsqueeze(z_y, 0)

            self.mu_y, self.logvar_y = self.netE_B.forward(z_y)
            std = self.logvar_y.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
            latent_code = eps.mul(std).add_(self.mu_y)
            feat_map = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
            	latent_code.size(0), latent_code.size(1), z_y.size(2), z_y.size(3))

            self.feat_map_zy = feat_map
            real_A_zy = []
            for i in range(0, self.opt.batchSize):
                _real = torch.unsqueeze(self.real_A[i], 0)
                _real = torch.cat((_real, feat_map), dim=1)
                real_A_zy.append(_real)
            real_A_zy = torch.cat(real_A_zy)
            self.real_A_zy.append(real_A_zy)

        os.chdir(self.origin_path)
        
        # feat_map for real images
        mu, logvar = self.netE_A(self.real_A)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        latent_code = eps.mul(std).add_(mu)
        self.real_A_feat_map = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
        	latent_code.size(0), latent_code.size(1), self.real_A.size(2), self.real_A.size(3))
         
        mu, logvar = self.netE_B(self.real_B)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        latent_code = eps.mul(std).add_(mu)
        self.real_B_feat_map = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
        	latent_code.size(0), latent_code.size(1), self.real_B.size(2), self.real_B.size(3))

    def inference(self):
        real_A = Variable(self.input_A).type(self.Tensor)
        real_B = Variable(self.input_B).type(self.Tensor)

        # feature map
        os.chdir(self.path_A)
        mc_sample_x = random.sample(self.list_A, 1)
        z_x = Image.open(mc_sample_x[0]).convert('RGB')
        z_x = self.img_resize(z_x, self.opt.loadSize)
        z_x = transform(z_x)
        if self.opt.input_nc == 1:  # RGB to gray
            z_x = z_x[0, ...] * 0.299 + z_x[1, ...] * 0.587 + z_x[2, ...] * 0.114
            z_x = z_x.unsqueeze(0)
        z_x = Variable(z_x).type(self.Tensor)
        z_x = torch.unsqueeze(z_x, 0)

        self.mu_x, self.logvar_x = self.netE_A.forward(z_x)
        std = self.logvar_x.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        latent_code = eps.mul(std).add_(self.mu_x)
        feat_map_zx = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
            latent_code.size(0), latent_code.size(1), z_x.size(2), z_x.size(3))

        os.chdir(self.path_B)
        mc_sample_y = random.sample(self.list_B, 1)
        z_y = Image.open(mc_sample_y[0]).convert('RGB')
        z_y = self.img_resize(z_y, self.opt.loadSize)
        z_y = transform(z_y)
        if self.opt.output_nc == 1:  # RGB to gray
            z_y = z_y[0, ...] * 0.299 + z_y[1, ...] * 0.587 + z_y[2, ...] * 0.114
            z_y = z_y.unsqueeze(0)
        z_y = Variable(z_y).type(self.Tensor)
        z_y = torch.unsqueeze(z_y, 0)
        
        self.mu_y, self.logvar_y = self.netE_B.forward(z_y)
        std = self.logvar_y.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        latent_code = eps.mul(std).add_(self.mu_y)
        feat_map_zy = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
        	latent_code.size(0), latent_code.size(1), z_y.size(2), z_y.size(3))

        os.chdir(self.origin_path)
       
        # feat_map for real images
        mu, logvar = self.netE_A(real_A)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        latent_code = eps.mul(std).add_(mu)
        real_A_feat_map = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
        	latent_code.size(0), latent_code.size(1), real_A.size(2), real_A.size(3))
        
        mu, logvar = self.netE_B(real_B)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
        latent_code = eps.mul(std).add_(mu)
        real_B_feat_map = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
        	latent_code.size(0), latent_code.size(1), real_B.size(2), real_B.size(3))

        # combine input image with random feature map
        real_B_zx = []
        for i in range(0, self.opt.batchSize):
            _real = torch.cat((real_B[i:i+1], feat_map_zx), dim=1)
            real_B_zx.append(_real)
        real_B_zx = torch.cat(real_B_zx)
        real_A_zy = []
        for i in range(0, self.opt.batchSize):
            _real = torch.cat((real_A[i:i+1], feat_map_zy), dim=1)
            real_A_zy.append(_real)
        real_A_zy = torch.cat(real_A_zy)

        # inference
        fake_B = self.netG_A(real_A_zy)
        fake_B_next = torch.cat((fake_B, real_A_feat_map), dim=1)
        self.rec_A = self.netG_B(fake_B_next).data
        self.fake_B = fake_B.data

        fake_A = self.netG_B(real_B_zx)
        fake_A_next = torch.cat((fake_A, real_B_feat_map), dim=1)
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

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        z = self.Tensor(batchSize, nz)
        if random_type == 'uni':
            z.copy_(torch.rand(batchSize, nz) * 2.0 - 1.0)
        elif random_type == 'gauss':
            z.copy_(torch.randn(batchSize, nz))
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
            _fake = fake_B[i*self.opt.batchSize:(i+1)*self.opt.batchSize]
            _fake = torch.cat((_fake, self.real_A_feat_map), dim=1)
            fake_B_next.append(_fake)
        fake_B_next = torch.cat(fake_B_next)

        rec_A = self.netG_B(fake_B_next)
        loss_cycle_A = 0
        for i in range(0, self.opt.mc_y):
            loss_cycle_A += self.criterionCycle(rec_A[i*self.opt.batchSize:(i+1)*self.opt.batchSize], self.real_A) * lambda_A
        pred_cycle_G_A = self.netD_B(rec_A)
        loss_cycle_G_A = self.criterionGAN(pred_cycle_G_A, True)

        # Backward cycle loss
        fake_A_next = []
        for i in range(0, self.opt.mc_x):
            _fake = fake_A[i*self.opt.batchSize:(i+1)*self.opt.batchSize]
            _fake = torch.cat((_fake, self.real_B_feat_map), dim=1)
            fake_A_next.append(_fake)
        fake_A_next = torch.cat(fake_A_next)

        rec_B = self.netG_A(fake_A_next)
        loss_cycle_B = 0
        for i in range(0, self.opt.mc_x):
            loss_cycle_B += self.criterionCycle(rec_B[i*self.opt.batchSize:(i+1)*self.opt.batchSize], self.real_B) * lambda_B
        pred_cycle_G_B = self.netD_A(rec_B)
        loss_cycle_G_B = self.criterionGAN(pred_cycle_G_B, True)

        # prior loss
        prior_loss_G_A = self.get_prior(self.netG_A.parameters(), self.opt.batchSize)
        prior_loss_G_B = self.get_prior(self.netG_B.parameters(), self.opt.batchSize)

        # KL loss
        kl_element = self.mu_x.pow(2).add_(self.logvar_x.exp()).mul_(-1).add_(1).add_(self.logvar_x)
        loss_kl_EA = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl

        kl_element = self.mu_y.pow(2).add_(self.logvar_y.exp()).mul_(-1).add_(1).add_(self.logvar_y)
        loss_kl_EB = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl

        # total loss
        loss_G =  loss_G_A + loss_G_B + (prior_loss_G_A + prior_loss_G_B) + (loss_cycle_G_A + loss_cycle_G_B) * self.opt.gamma + (loss_cycle_A + loss_cycle_B) + loss_kl_EA + loss_kl_EB
        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data

        self.loss_G_A = loss_G_A.data[0] + loss_cycle_G_A.data[0] + prior_loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0] + loss_cycle_G_B.data[0] + prior_loss_G_A.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_kl_EA = loss_kl_EA.data[0]
        self.loss_kl_EB = loss_kl_EB.data[0]

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

    def optimize(self):
        # forward
        self.forward()
        # G_A and G_B
        # E_A and E_B
        self.optimizer_G.zero_grad()
        self.optimizer_E_A.zero_grad()
        self.optimizer_E_B.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_E_A.step()
        self.optimizer_E_B.step()
        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
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
        elif self.opt.gamma > 0 or self.opt.gamma < 1:
            loss['cyc_G_A'] = self.loss_cycle_A
            loss['cyc_G_B'] = self.loss_cycle_B
        if self.opt.lambda_kl > 0:
        	loss['kl_EA'] = self.loss_kl_EA
        	loss['kl_EB'] = self.loss_kl_EB
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
            prior_loss += torch.mean(param*param)
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
        self.save_network(self.netE_A, 'E_A', label)
        self.save_network(self.netE_B, 'E_B', label)
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
            lr = self.opt.lr * np.exp(-1.0 * min(1.0, epoch_iter/float(dataset_size)))
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

