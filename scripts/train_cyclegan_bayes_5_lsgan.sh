nohup python train_bayes.py --dataroot /root/nfs-datasets/cityscapes --name cityscapes_bayes_5_lsgan --batchSize 1 --loadSize 256 --ratio 2 --netG_A global --netG_B global --ngf 32 --num_D_A 1 --num_D_B 1 --mc_x 3 --mc_y 3 --n_blocks_global 6 --n_downsample_global 2 --niter 50 --niter_decay 50 --gamma 0 --lambda_kl 0.1 --continue_train --which_epoch 40000 >>cityscapes_bayes_5_lsgan.out &