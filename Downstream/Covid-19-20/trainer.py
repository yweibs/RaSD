# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import AverageMeter, distributed_all_gather

from monai.data import decollate_batch

import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
from monai.visualize import plot_2d_or_3d_image
import torchvision.utils as vutils
from datetime import datetime

def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(), target.cuda()
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            #
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
        
        lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()

        length = len(loader) // 4
        if args.rank == 0 and (idx + 1) % length == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "lr: {:.8f}".format(lr),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    
    return run_loss.avg


# def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
#     model.eval()
#     run_acc = AverageMeter()
#     start_time = time.time()
#     num = np.zeros(args.out_channels - 1)

#     all_dice = None

#     with torch.no_grad():
#         for idx, batch_data in enumerate(loader):
#             data, target = batch_data["image"], batch_data["label"]
#             data, target = data.cuda(), target.cuda()

#             with autocast(enabled=args.amp):
#                 if model_inferer is not None:
#                     logits = model_inferer(data)
#                 else:
#                     logits = model(data)
#             if not logits.is_cuda:
#                 target = target.cpu()

#             val_labels_list = decollate_batch(target)
#             val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
#             val_outputs_list = decollate_batch(logits)
#             val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
#             acc_func.reset()
#             acc_func(y_pred=val_output_convert, y=val_labels_convert)
#             acc, not_nans = acc_func.aggregate()
#             acc = acc.cuda(args.rank)

#             dice_list_sub = []
#             val_outputs = logits.argmax(1)[0].data.cpu().numpy().astype(np.uint8)
#             val_labels = target.data.cpu().numpy()[0, 0, :, :, :]
#             from utils.utils import dice

#             for i in range(1, args.out_channels):
#                 num[i - 1] += (np.sum(val_labels == i) > 0).astype(np.uint8)
#                 organ_Dice = dice(val_outputs == i, val_labels == i)
#                 dice_list_sub.append(organ_Dice)
#             # print("Organ and Tumor Dice:", dice_list_sub)

#             if all_dice is None:
#                 all_dice = (np.asarray(dice_list_sub)).copy()
#             else:
#                 all_dice = all_dice + np.asarray(dice_list_sub)
#             # print("Organ and Tumor Dice accumulate:", (all_dice / num))

#             if args.distributed:
#                 acc_list, not_nans_list = distributed_all_gather(
#                     [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
#                 )
#                 for al, nl in zip(acc_list, not_nans_list):
#                     run_acc.update(al, n=nl)

#             else:
#                 run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

#             if args.rank == 0:
#                 avg_acc = np.mean(run_acc.avg)

#                 print(
#                     "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
#                     "acc",
#                     avg_acc,
#                     "time {:.2f}s".format(time.time() - start_time),
#                 )
#             start_time = time.time()
#     torch.cuda.empty_cache()
#     tumor_acc = all_dice / num
#     tumor_acc = tumor_acc[-1]
#     return tumor_acc

def val_epoch(model, loader, epoch, acc_func, args, model_inferer=None, post_label=None, post_pred=None):
    model.eval()
    run_acc = AverageMeter()
    start_time = time.time()
    num = np.zeros(args.out_channels - 1)

    all_dice = None
    
    # 创建可视化目录
    if args.rank == 0 and args.visualize and (epoch % args.visualize_freq == 0 or epoch == args.max_epochs - 1):
        vis_dir = os.path.join(args.logdir, "visualization")
        os.makedirs(vis_dir, exist_ok=True)
        sample_count = 0  # 计数保存的样本数量

    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            data, target = batch_data["image"], batch_data["label"]
            data, target = data.cuda(), target.cuda()

            with autocast(enabled=args.amp):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)
            if not logits.is_cuda:
                target = target.cpu()

            val_labels_list = decollate_batch(target)
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            acc_func.reset()
            acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc, not_nans = acc_func.aggregate()
            acc = acc.cuda(args.rank)

            dice_list_sub = []
            val_outputs = logits.argmax(1)[0].data.cpu().numpy().astype(np.uint8)
            val_labels = target.data.cpu().numpy()[0, 0, :, :, :]
            from utils.utils import dice

            for i in range(1, args.out_channels):
                num[i - 1] += (np.sum(val_labels == i) > 0).astype(np.uint8)
                organ_Dice = dice(val_outputs == i, val_labels == i)
                dice_list_sub.append(organ_Dice)
            # print("Organ and Tumor Dice:", dice_list_sub)

            if all_dice is None:
                all_dice = (np.asarray(dice_list_sub)).copy()
            else:
                all_dice = all_dice + np.asarray(dice_list_sub)
            # print("Organ and Tumor Dice accumulate:", (all_dice / num))
            
            # =============== 添加可视化代码 ===============
            if args.rank == 0 and args.visualize and (epoch % args.visualize_freq == 0 or epoch == args.max_epochs - 1) and sample_count < args.max_visualize:
                # 只可视化前几个样本
                for i in range(min(data.size(0), args.max_visualize - sample_count)):
                    # 获取中间切片
                    slice_idx = data.shape[2] // 2  # 3D数据的中间切片
                    
                    # 原始图像
                    img_slice = data[i, 0, slice_idx].cpu().numpy()  # 假设单通道，取第一个通道
                    
                    # 真实标签
                    label_slice = target[i, 0, slice_idx].cpu().numpy()
                    
                    # 预测结果
                    pred_slice = torch.argmax(logits[i], dim=0)[slice_idx].cpu().numpy()
                    
                    # 创建可视化图形
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # 原始图像
                    axes[0].imshow(img_slice, cmap='gray')
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    
                    # 真实标签
                    axes[1].imshow(label_slice, cmap='jet')
                    axes[1].set_title('Ground Truth')
                    axes[1].axis('off')
                    
                    # 预测结果
                    axes[2].imshow(pred_slice, cmap='jet')
                    axes[2].set_title('Prediction')
                    axes[2].axis('off')
                    
                    # 保存图像
                    plt.suptitle(f'Epoch: {epoch}, Sample: {sample_count}, Slice: {slice_idx}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, f'epoch_{epoch}_sample_{sample_count}_slice_{slice_idx}.png'), 
                               dpi=150, bbox_inches='tight')
                    plt.close(fig)
                    
                    sample_count += 1
            # =============== 结束可视化代码 ===============

            if args.distributed:
                acc_list, not_nans_list = distributed_all_gather(
                    [acc, not_nans], out_numpy=True, is_valid=idx < loader.sampler.valid_length
                )
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(al, n=nl)

            else:
                run_acc.update(acc.cpu().numpy(), n=not_nans.cpu().numpy())

            if args.rank == 0:
                avg_acc = np.mean(run_acc.avg)

                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "acc",
                    avg_acc,
                    "time {:.2f}s".format(time.time() - start_time),
                )
            start_time = time.time()
    torch.cuda.empty_cache()
    tumor_acc = all_dice / num
    tumor_acc = tumor_acc[-1]
    return tumor_acc


def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    model_inferer=None,
    scheduler=None,
    start_epoch=0,
    post_label=None,
    post_pred=None,
):
    if not hasattr(args, 'visualize'):
        args.visualize = True  # 是否启用可视化
    if not hasattr(args, 'visualize_freq'):
        args.visualize_freq = 5  # 每5个epoch可视化一次
    if not hasattr(args, 'max_visualize'):
        args.max_visualize = 4  # 最多可视化4个样本
        
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc = val_epoch(
                model,
                val_loader,
                epoch=epoch,
                acc_func=acc_func,
                model_inferer=model_inferer,
                args=args,
                post_label=post_label,
                post_pred=post_pred,
            )

            val_avg_acc = np.mean(val_avg_acc)

            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )
            if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                save_checkpoint(model, epoch, args, best_acc=val_acc_max, filename="model_final.pt")
                if b_new_best:
                    print("Copying to model.pt new best model!!!!")
                    shutil.copyfile(os.path.join(args.logdir, "model_final.pt"), os.path.join(args.logdir, "model.pt"))

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
