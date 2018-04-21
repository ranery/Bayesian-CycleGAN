
# -*- coding: utf-8 -*-
import os
import glob
import numpy as np
from cv2 import imread

def analyze_div(X_real, X_sample):
    
    def kl_div(p, q):
        eps = 1e-10
        p_safe = np.copy(p)
        p_safe[p_safe < eps] = eps
        q_safe = np.copy(q)
        q_safe[q_safe < eps] = eps
        return np.sum(p_safe * (np.log(p_safe) - np.log(q_safe)))

    def js_div(p, q):
        m = (p + q) / 2.
        return (kl_div(p, m) + kl_div(q, m))/2.
    
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_trans_real = pca.fit_transform(X_real)
    X_trans_fake = pca.transform(X_sample)
    
    from scipy import stats

    def cartesian_prod(x, y):
        return np.dstack(np.meshgrid(x, y)).reshape(-1, 2)

    dx = 0.1
    dy = 0.1
    
    xmin1 = np.min(X_trans_real[:, 0]) - 3.0
    xmax1 = np.max(X_trans_real[:, 0]) + 3.0
    
    xmin2 = np.min(X_trans_real[:, 1]) - 3.0
    xmax2 = np.max(X_trans_real[:, 1]) + 3.0
    
    space = cartesian_prod(np.arange(xmin1,xmax1,dx), np.arange(xmin2,xmax2,dy)).T

    real_kde = stats.gaussian_kde(X_trans_real.T)
    real_density = real_kde(space) * dx * dy

    fake_kde = stats.gaussian_kde(X_trans_fake.T)
    fake_density = fake_kde(space) * dx * dy
    

    return js_div(real_density, fake_density), X_trans_real, X_trans_fake


if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', required=True, help='real samples')
    parser.add_argument('--fake_dir', required=True, help='fake samples')
    opt = parser.parse_args()
    real_names = glob.glob(os.path.join(opt.real_dir, '*.png'))
    real_names += glob.glob(os.path.join(opt.real_dir, '*.jpg'))
    real_images = np.array([imread(str(fn)).astype(np.float32).reshape(-1) for fn in real_names])
    fake_names = glob.glob(os.path.join(opt.fake_dir, '*.png'))
    fake_names += glob.glob(os.path.join(opt.fake_dir, '*.jpg'))
    fake_images = np.array([imread(str(fn)).astype(np.float32).reshape(-1) for fn in fake_names])

    m = analyze_div(real_images, fake_images)

    print(m)




