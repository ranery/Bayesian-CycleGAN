# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""
import numpy as np
import torch
import os
from torch.autograd import Variable
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.append('/home/chengyu/hryou/code/cyclegan/')
from options.train_options import TrainOptions
from models import networks
import util.util as util
import random
from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

class Encoder():
    def name(self):
        return 'Encoder model'

    def initialize(self, opt):
        self.opt = opt
        if torch.cuda.is_available():
            print('cuda is available, we will use gpu!')
            self.Tensor = torch.cuda.FloatTensor
            torch.cuda.manual_seed_all(100)
        else:
            self.Tensor = torch.FloatTensor
            torch.manual_seed(100)
        self.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.save_dir = os.path.join(opt.checkpoints_dir, 'features')

        # load network
        self.netE_A = networks.define_G(opt.input_nc, 8, 64, 'encoder', norm=opt.norm).type(self.Tensor)
        self.netE_B = networks.define_G(opt.output_nc, 8, 64, 'encoder', norm=opt.norm).type(self.Tensor)
        self.load_network(self.netE_A, 'E_A', opt.which_epoch, self.load_dir)
        self.load_network(self.netE_B, 'E_B', opt.which_epoch, self.load_dir)

        # dataset path and name list
        self.origin_path = os.getcwd()
        self.path_A = self.opt.dataroot + '/trainA'
        self.path_B = self.opt.dataroot + '/trainB'
        self.list_A = os.listdir(self.path_A)
        self.list_B = os.listdir(self.path_B)

    def get_feature(self):
        for img in self.list_A:
            os.chdir(self.path_A)
            A = Image.open(img).convert('RGB')
            A = transform(model.img_resize(A, opt.loadSize))
            if self.opt.input_nc == 1:  # RGB to gray
                A = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
                A = A.unsqueeze(0)
            A = Variable(torch.unsqueeze(A, 0)).type(self.Tensor)

            mu_x, logvar_x = self.netE_A.forward(A)
            std = logvar_x.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
            latent_code = eps.mul(std).add_(mu_x)
            feat_A = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
                latent_code.size(0), latent_code.size(1), A.size(2), A.size(3))
            feat_A = feat_A.data
            print(feat_A)

            # feat_A = util.tensor2im(feat_A)
            # img_path = os.path.join(self.save_dir, 'trainA/%s' % img)
            # os.chdir(self.origin_path)
            # util.save_image(feat_A, img_path)
        
        for img in self.list_B:
            os.chdir(self.path_B)
            B = Image.open(img).convert('RGB')
            B = transform(model.img_resize(B, opt.loadSize))
            if self.opt.input_nc == 1:  # RGB to gray
                B = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
                B = B.unsqueeze(0)
            B = Variable(torch.unsqueeze(B, 0)).type(self.Tensor)

            mu_y, logvar_y = self.netE_B.forward(B)
            std = logvar_y.mul(0.5).exp_()
            eps = self.get_z_random(std.size(0), std.size(1), 'gauss')
            latent_code = eps.mul(std).add_(mu_y)
            feat_B = latent_code.view(latent_code.size(0), latent_code.size(1), 1, 1).expand(
                latent_code.size(0), latent_code.size(1), B.size(2), B.size(3))
            feat_B = feat_B.data
            
            # feat_B = util.tensor2im(feat_B)
            # img_path = os.path.join(self.save_dir, 'trainB/%s' % img)
            # os.chdir(self.origin_path)
            # util.save_image(feat_B, img_path)

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

    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.load_dir, save_filename)
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

if __name__ == '__main__':
    model = Encoder()
    opt = TrainOptions().parse()
    model.initialize(opt)
    model.get_feature()