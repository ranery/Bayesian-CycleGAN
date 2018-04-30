import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--name', type=str, required=True, help='experiment name')
parser.add_argument('--data_dir', type=str, required=True, help='dataset dir')
parser.add_argument('--bayes', action='store_true', help='use bayes')
parser.add_argument('--howmany', type=int, default=500, help='test images num')
parser.add_argument('--epoch', type=int, default=10, help='which epoch to test')
parser.add_argument('--dataset_name', type=str, required=True, help='dataset nane to distinguish')

opt = parser.parse_args()
if opt.bayes:
    expr_phase = 'CycleGAN_Bayes'
    test = 'test_bayes.py'
else:
    expr_phase = 'CycleGAN'
    test = 'test.py'


print("[Test] test model: {}".format(expr_phase))


command = '/root/workspace/ccc/anaconda3/bin/python {}'.format(test) + ' --dataroot {}'.format(opt.data_dir) +\
          ' --phase test --loadSize 512  --netG_A global --netG_B global' +\
          ' --ngf 32 --n_blocks_global 8 --n_downsample_global 2' +\
          ' --which_epoch {} --how_many {}'.format(opt.epoch, opt.howmany) + \
          ' --name {}'.format(opt.name)

# os.system(command)

print("----------------------------------------------------------------------")
print("[Extract] extract images from generated dir")

base_dir = '/root/workspace/ccc/Bayesian-CycleGAN/results/{}/test_{}/'.format(opt.name, opt.epoch)
result_dir = base_dir + 'images/'
save_dir = base_dir
command = '/root/workspace/ccc/anaconda3/bin/python datasets/extract.py' + ' --dir {} --save_dir {}'.format(result_dir, save_dir)
command1 = command + ' --type fake_A'
command2 = command + ' --type fake_B'

os.system(command1)
os.system(command2)

print("----------------------------------------------------------------------")
print("[Inception Score] calculate inception score")

command = '/root/workspace/ccc/anaconda3/bin/python evaluation/precalc_stats_inception.py'
fake_A_dir = base_dir + 'fake_A/'
fake_B_dir = base_dir + 'fake_B/'
command1 = command + ' --dir {}'.format(fake_A_dir)
command2 = command + ' --dir {}'.format(fake_B_dir)

os.system(command1)
os.system(command2)

print("----------------------------------------------------------------------")
print("[FID Score] calculate fid score")

test_base_dir = 'test_result/'
if not os.path.exists(test_base_dir):
    os.mkdir(test_base_dir)

if not os.path.exists(test_base_dir + opt.dataset_name):
    os.mkdir(test_base_dir + opt.dataset_name)

fid_result_dir = test_base_dir + opt.dataset_name +'/fid/'

if not os.path.exists(fid_result_dir):
    os.mkdir(fid_result_dir)

if not os.path.exists(fid_result_dir+'test_A.npz'):
    print("test_A.npz not found, create it")
    command = '/root/workspace/ccc/anaconda3/bin/python evaluation/precalc_stats_fid.py --data_dir {} --out_dir {}'\
        .format(opt.data_dir+'testA/', fid_result_dir+'test_A.npz')
    os.system(command)

if not os.path.exists(fid_result_dir+'test_B.npz'):
    print("test_B.npz not found, create it")
    command = '/root/workspace/ccc/anaconda3/bin/python evaluation/precalc_stats_fid.py --data_dir {} --out_dir {}'\
        .format(opt.data_dir+'testB/', fid_result_dir+'test_B.npz')
    os.system(command)

if not os.path.exists(fid_result_dir+'train_A.npz'):
    print("train_A.npz not found, create it")
    command = '/root/workspace/ccc/anaconda3/bin/python evaluation/precalc_stats_fid.py --data_dir {} --out_dir {}'\
        .format(opt.data_dir+'trainA/', fid_result_dir+'train_A.npz')
    os.system(command)

if not os.path.exists(fid_result_dir+'train_B.npz'):
    print("train_B.npz not found, create it")
    command = '/root/workspace/ccc/anaconda3/bin/python evaluation/precalc_stats_fid.py --data_dir {} --out_dir {}'\
        .format(opt.data_dir+'trainB/', fid_result_dir+'train_B.npz')
    os.system(command)

command3 = '/root/workspace/ccc/anaconda3/bin/python evaluation/precalc_stats_fid.py --data_dir={} --out_dir={}'\
    .format(fake_B_dir, fid_result_dir+'{}_fake_B.npz'.format(opt.name))

os.system(command3)

command4 = '/root/workspace/ccc/anaconda3/bin/python evaluation/precalc_stats_fid.py --data_dir={} --out_dir={}'\
    .format(fake_A_dir, fid_result_dir+'{}_fake_A.npz'.format(opt.name))

os.system(command4)


print("[FID SCORE] Test B and Fake B")
command5 = '/root/workspace/ccc/anaconda3/bin/python evaluation/fid.py {}test_B.npz {}{}_fake_B.npz  --gpu 0'.format(fid_result_dir, fid_result_dir, opt.name)
os.system(command5)

print("[FID SCORE] Test A and Fake A")
command6 = '/root/workspace/ccc/anaconda3/bin/python evaluation/fid.py {}test_A.npz {}{}_fake_A.npz  --gpu 0'.format(fid_result_dir, fid_result_dir,opt.name)
os.system(command6)

print("[FID SCORE] Train B and Fake B")
command7 = '/root/workspace/ccc/anaconda3/bin/python evaluation/fid.py {}train_B.npz {}{}_fake_B.npz  --gpu 0'.format(fid_result_dir, fid_result_dir, opt.name)
os.system(command7)

print("[FID SCORE] Train A and Fake A")
command7 = '/root/workspace/ccc/anaconda3/bin/python evaluation/fid.py {}train_A.npz {}{}_fake_A.npz  --gpu 0'.format(fid_result_dir, fid_result_dir,opt.name)
os.system(command7)

