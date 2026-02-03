import random
from os.path import join
from os import listdir
import SimpleITK as sitk
import torch
from torch.utils import data
import numpy as np
from torch.utils.data import DataLoader

import torch.nn.functional as F
from utils.STN import AffineTransformer, SpatialTransformer


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

class Memory(object):
    def __init__(self, lenth):
        self.lenth = lenth
        self.img = []
        self.lab = []
    def push(self, img_one, lab_one):
        self.img.append(img_one)
        self.lab.append(lab_one)

    def pop(self):
        self.img.pop(0)
        self.lab.pop(0)

    def update(self, img_one, lab_one):
        self.pop()
        self.push(img_one, lab_one)

    def get(self):
        img = torch.cat(self.img, dim=0)
        lab = torch.cat(self.lab, dim=0)
        return img, lab

    def init_mem(self, img_one, lab_one):
        self.push(img_one, lab_one)
        if len(self.img) < self.lenth:
            return True
        else:
            return False

class DatasetFromFolder3D(data.Dataset):
    def __init__(self, file_dir):
        super(DatasetFromFolder3D, self).__init__()
        self.filenames = [x for x in listdir(join(file_dir, 'Image')) if is_image_file(x)]
        self.file_dir = file_dir

    def __getitem__(self, index):
        img = sitk.ReadImage(join(self.file_dir, 'Image', self.filenames[index]))
        img = sitk.GetArrayFromImage(img)
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, :]

        lab = sitk.ReadImage(join(self.file_dir, 'Label', self.filenames[index]))
        lab = sitk.GetArrayFromImage(lab)
        lab = lab.astype(np.float32)
        lab = self.to_categorical(lab)

        return img, lab

    def to_categorical(self, y, num_classes=None):
        y = np.array(y, dtype='int')
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        if not num_classes:
            num_classes = np.max(y) + 1
        n = y.shape[0]
        categorical = np.zeros((num_classes, n))
        categorical[y, np.arange(n)] = 1
        output_shape = (num_classes,) + input_shape
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def __len__(self):
        return len(self.filenames)

class DatasetGEN3D_CacheDataloader():
    def __init__(self, file_dir, batchsize, bagsize=16, baglen=1):
        super(DatasetGEN3D_CacheDataloader, self).__init__()
        self.batchsize = batchsize
        self.atn = AffineTransformer()
        self.stn = SpatialTransformer()
        self.filenames = [x for x in listdir(join(file_dir, 'Image')) if is_image_file(x)]
        self.file_dir = file_dir

        self.is_init = True

        self.bag_mem = Memory(baglen)
        self.data_loader = DataLoader(DatasetFromFolder3D(file_dir), batch_size=bagsize, shuffle=True)
        self.indexs_list = list(range(bagsize*baglen))

        # Initialize
        self.init()

    def init(self):
        if self.is_init:
            init = True
            while init:
                img, lab = next(self.data_loader.__iter__())

                init = self.bag_mem.init_mem(img, lab)
            self.update_data_pool()

            self.is_init = False

    def update_bag_mem(self):
        img, lab = next(self.data_loader.__iter__())

        self.bag_mem.update(img, lab)

    def update_data_pool(self):
        self.img_pool, self.lab_pool = self.bag_mem.get()

    def __getitem__(self):
        random.shuffle(self.indexs_list)#[0:self.batchsize]
        indexs = self.indexs_list[0:self.batchsize]

        img = self.img_pool[indexs]
        lab = self.lab_pool[indexs]

        return img, lab
