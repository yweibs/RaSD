import torch
import numpy as np
import torch.nn.functional as F

from utils.STN import AffineTransformer, SpatialTransformer
# from utils.Transform_torch_ import SpatialTransform
from utils.gen_img import labels_to_image
from utils.gen_lab import draw_perlin


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".nii"])

def imgnorm(N_I, index1=0.015, index2=0.015):
    I_sort = np.sort(N_I.flatten())
    I_min = I_sort[int(index1 * len(I_sort))]
    I_max = I_sort[-int(index2 * len(I_sort))]

    N_I = 1.0 * (N_I - I_min) / (I_max - I_min+1e-6)
    N_I[N_I > 1.0] = 1.0
    N_I[N_I < 0.0] = 0.0
    N_I2 = N_I.astype(np.float32)
    return N_I2

def limit(image):
    max = np.where(image < 0)
    image[max] = 0
    return image

def Nor(data):
    data = np.asarray(data)
    min = np.min(data)
    max = np.max(data)
    data = (data - min) / (max - min)
    return data

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

class DatasetGEN3D():
    def __init__(self, in_shape=(256,),
                 num_label=16,
                 batchsize=1):
        super(DatasetGEN3D, self).__init__()
        self.in_shape = in_shape * 2
        self.batchsize = batchsize
        self.num_dim = len(self.in_shape)
        self.num_label = num_label
        self.stn = SpatialTransformer()

        self.step = 1 if self.batchsize//10 == 0 else self.batchsize//10
        self.is_init=True

        self.data_mem = Memory(self.batchsize)

    def init(self):
        init = True
        while init:
            is_nan = True
            while is_nan:
                img_one, lab_one, nan_eval = self.gen_data()
                is_nan = nan_eval
            init = self.data_mem.init_mem(img_one, lab_one)
        self.is_init=False

    def gen_lab_img(self):
        lab = draw_perlin(out_shape=(self.num_label, 1, *self.in_shape),
                          scales=(self.in_shape[0]//8, self.in_shape[0]//4),
                          max_std=1,
                          device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        flow = draw_perlin(out_shape=(self.num_label, self.num_dim, *self.in_shape),
                           scales=(self.in_shape[0]//16, self.in_shape[0]//8, self.in_shape[0]//4),
                           max_std=self.in_shape[0]//16,
                           device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        lab = self.stn(lab, flow)

        lab = torch.argmax(lab, dim=0)[:, np.newaxis].type(torch.float32)
        img = labels_to_image(lab, self.num_label, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return img, lab
    def gen_data(self):
        img, lab = self.gen_lab_img()
        lab = lab.type(torch.int64)
        lab = F.one_hot(lab[:, 0], self.num_label).permute(0, -1, 1, 2)
        img = img.type(torch.float32)
        if torch.any(torch.isnan(img)) or torch.any(torch.isinf(img)):
            is_nan = True
        else:
            is_nan = False
        return img, lab, is_nan

    # def to_categorical(self, y, num_classes=None):
    #     y = np.array(y, dtype='int')
    #     input_shape = y.shape
    #     if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
    #         input_shape = tuple(input_shape[:-1])
    #     y = y.ravel()
    #     if not num_classes:
    #         num_classes = np.max(y) + 1
    #     n = y.shape[0]
    #     categorical = np.zeros((num_classes, n))
    #     categorical[y, np.arange(n)] = 1
    #     output_shape = (num_classes,) + input_shape
    #     categorical = np.reshape(categorical, output_shape)
    #     return categorical

    def __getitem__(self):
        if self.is_init:
            self.init()
        for _ in range(self.step):
            is_nan = True
            while is_nan:
                img_one, lab_one, nan_eval = self.gen_data()
                is_nan = nan_eval
            self.data_mem.update(img_one, lab_one)
        img, lab = self.data_mem.get()

        return img, lab


