import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize':         # 1024 x 1024
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        # transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':  # 1024 x 512
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    else:
        w = target_width
        h = int(target_width * oh / ow)
    """
    if target_width == 0:
        # for trian (large scale dataset)
        if ow > 1800 or oh > 1800:
            ow = ow / 6
            oh = oh / 6
        elif ow > 1400 or oh > 1400:
            ow = ow / 5
            oh = oh / 5
        elif ow > 1000 or oh > 1000:
            ow = ow / 4
            oh = oh / 4
        elif ow > 600 or oh > 600:
            ow = ow / 2
            oh = oh / 2
        
        # for test (large scale dataset)
        if ow > 1600 or oh > 1600:
            ow = ow / 3
            oh = oh / 3
        if ow > 1000 or oh > 1000:
            ow = ow / 1.5
            oh = oh / 1.5
        
        for index in range(100):
            low = index * 16
            high = (index + 1) * 16
            if ow > low and ow <= high:
                w = high
            if oh > low and oh <= high:
                h = high
    else:
        w = target_width
        h = int(target_width * oh / ow)
    """
    # print('(%d, %d)' % (w, h))
    return img.resize((w, h), Image.BICUBIC)
