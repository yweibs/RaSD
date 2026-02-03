import os
from time import time

import torch
from torch import nn
from utils.dataloader_gen_online import DatasetGEN3D
from utils.STN import SpatialTransformer
from utils.losses import gradient_loss, MSE, ncc_loss, dice_loss, MAE, partical_MAE, crossentropy
from utils.utils import AverageMeter, to_categorical
from monai.networks.nets import SwinUNETR
import numpy as np
import SimpleITK as sitk
import time

class Trainer(object):
    def __init__(self, k=0,
                 n_channels=1,
                 lr=1e-5,
                 epoches=1,
                 iters=100,
                 batch_size=1,
                 checkpoint_dir='weights',
                 result_dir='results',
                 model_name='RaSD'
                 ):
        super(Trainer, self).__init__()

        self.k = k
        self.epoches = epoches
        self.iters = iters
        self.lr = lr
        self.shape = 128
        self.num_label = self.shape // 16

        self.results_dir = result_dir
        self.checkpoint_dir = checkpoint_dir
        self.model_name = model_name

        # self.Network = XMorpher_Plus(in_chans=n_channels)

        self.Network = SwinUNETR(
            img_size=self.shape,
            in_channels=n_channels,
            out_channels=32,
            feature_size=48,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            dropout_path_rate=0.0,
            use_checkpoint=True,
            use_v2=True
        )

        if torch.cuda.is_available():
            self.Network = self.Network.cuda()
        self.opt = torch.optim.AdamW(self.Network.parameters(), lr=lr)

        self.stn = SpatialTransformer()
        self.train_dataset = DatasetGEN3D(in_shape=(self.shape,),
                                           num_label=self.shape // 16,
                                           batchsize=batch_size,)


        self.L_ce = crossentropy

        self.L_prototypes_log = AverageMeter(name='L_prototypes')
        self.L_seg_log = AverageMeter(name='L_seg')

    def train_iterator(self, img, lab, t=0.07):
        f = self.Network(img)

        prototypes = []
        for i in range(self.num_label):
            prototypes.append(torch.sum(f*lab[:, i:i+1], dim=(2, 3, 4), keepdim=True)/torch.sum(lab[:, i:i+1], dim=(2, 3, 4), keepdim=True))

        seg = []
        for i in range(self.num_label):
            # a = torch.cosine_similarity(prototypes_mean[i], f, dim=1)[:, np.newaxis, :, :, :]
            seg.append(torch.sum(prototypes[i] * f, dim=1, keepdim=True))

        # prototypes_var = []
        # for i in range(self.num_label):
        #     prototypes_var.append(torch.sum(torch.pow(f - prototypes_mean[i], 2)*lab[:, i:i+1], dim=(2, 3, 4), keepdim=True)/torch.sum(lab[:, i:i+1], dim=(2, 3, 4), keepdim=True))

        seg = torch.cat(seg, dim=1)
        prototypes = torch.cat(prototypes, dim=2)[:, :, :, 0, 0]

        y_pred_prototypes = prototypes.transpose(-2, -1) @ prototypes
        y_pred_prototypes = torch.softmax(y_pred_prototypes/t, dim=1)

        seg = torch.softmax(seg/t, dim=1)

        y_true = torch.eye(self.num_label, device=y_pred_prototypes.device)[np.newaxis].type(torch.float32)

        loss_prototypes = self.L_ce(y_pred_prototypes, y_true)
        loss_seg = self.L_ce(seg, lab)
        loss = loss_prototypes + loss_seg
        loss.backward()
        self.opt.step()
        self.Network.zero_grad()
        self.opt.zero_grad()

        self.L_prototypes_log.update(loss_prototypes.data, img.size(0))
        self.L_seg_log.update(loss_seg.data, img.size(0))

    def train_epoch(self, epoch):
        self.Network.train()
        # data_start_time = time.time()
        for i in range(self.iters):
            img, lab = self.train_dataset.__getitem__()

            if torch.cuda.is_available():
                img = img.cuda()
                lab = lab.cuda()

            self.train_iterator(img, lab)
            res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, self.epoches),
                             'Iter: [%d/%d]' % (i + 1, self.iters),
                             self.L_prototypes_log.__str__(),
                             self.L_seg_log.__str__()])
            print(res)
        # data_read_time = time.time() - data_start_time
        # print("data_read_time", data_read_time)

    def checkpoint(self, epoch):
        torch.save(self.Network.state_dict(),
                   '{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, epoch+self.k))

    def load(self):
        self.Network.load_state_dict(
            torch.load('{0}/{1}_epoch_{2}.pth'.format(self.checkpoint_dir, self.model_name, str(self.k))))

    def train(self):
        for epoch in range(self.epoches-self.k):
            self.L_prototypes_log.reset()
            self.L_seg_log.reset()

            self.train_epoch(epoch+self.k)
            if epoch % 20 == 0:
                self.checkpoint(epoch)
        self.checkpoint(self.epoches-self.k)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Online pretraining using generator')
    parser.add_argument('--n_channels', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epoches', type=int, default=10000)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='weights')
    parser.add_argument('--result_dir', type=str, default='results')
    parser.add_argument('--model_name', type=str, default='RaSD')
    parser.add_argument('--shape', type=int, default=128)
    parser.add_argument('--k', type=int, default=0, help='start epoch offset')
    parser.add_argument('--load', action='store_true', help='load checkpoint before training')
    parser.add_argument('--cuda_devices', type=str, default='0', help='CUDA_VISIBLE_DEVICES')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_devices
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    start_time = time.time()
    trainer = Trainer(k=args.k,
                      n_channels=args.n_channels,
                      lr=args.lr,
                      epoches=args.epoches,
                      iters=args.iters,
                      batch_size=args.batch_size,
                      checkpoint_dir=args.checkpoint_dir,
                      result_dir=args.result_dir,
                      model_name=args.model_name)

    if args.load:
        try:
            trainer.load()
            print('Loaded checkpoint successfully')
        except Exception as e:
            print(f'Warning: failed to load checkpoint: {e}')

    trainer.train()
    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")
    # trainer.test()