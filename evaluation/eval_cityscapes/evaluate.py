import os
import sys
import argparse
import numpy as np
import scipy.misc
from PIL import Image
from util import *
from cityscapes import cityscapes

parser = argparse.ArgumentParser()
parser.add_argument("--cityscapes_dir", type=str, required=True, help="Path to the original cityscapes dataset")
parser.add_argument("--result_dir", type=str, required=True, help="Path to the generated images to be evaluated")
args = parser.parse_args()

def main():

    CS = cityscapes(args.cityscapes_dir)
    n_cl = len(CS.classes)

    origin_images = os.listdir(args.cityscapes_dir)
    test_images = os.listdir(args.result_dir)

    hist_perframe = np.zeros((n_cl, n_cl))

    for i, idx in enumerate(test_images):
        if i % 10 == 0:
            print('Evaluating: %d/%d' % (i, len(test_images)))
        image_name = origin_images[i].replace('_gtFine_color.png', '') #.split('.')[0]
        gen_image_name = image_name + '_leftImg8bit_fake_B.png' #+ '_fake_B.png'

        label = CS.load_label(os.path.join(args.cityscapes_dir, origin_images[i]))
        out = CS.load_label(os.path.join(args.result_dir, gen_image_name)) # np.array(Image.open(os.path.join(args.result_dir, gen_image_name)) #.convert('RGB'))
        hist_perframe += fast_hist(label.flatten(), out.flatten(), n_cl)
    # print hist_perframe
    mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
    with open('./evaluation_results.txt', 'w') as f:
        f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
        f.write('Mean class accuracy: %f\n' % mean_class_acc)
        f.write('Mean class IoU: %f\n' % mean_class_iou)
        f.write('************ Per class numbers below ************\n')
        for i, cl in enumerate(CS.classes):
            while len(cl) < 15:
                cl = cl + ' '
            f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))
main()
