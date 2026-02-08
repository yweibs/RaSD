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
import torch.nn.functional as F
from monai.data import decollate_batch
import matplotlib.pyplot as plt
from sklearn import metrics


def resize(img):
    size = 256
    b, c, h, w = img.size()
    # print(img.shape)
    new_img = []
    # for i in range(b):
        # im = img[i:i+1, :, :, :]
    img = F.interpolate(img, size=[size, size], mode='bilinear', align_corners=True)
        # new_img.append(im.unsqueeze(0))
    # new_img = torch.cat(new_img, dim=0)
    return img


def train_epoch(model, loader, optimizer, scheduler, scaler, epoch, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()

    loss_func = torch.nn.CrossEntropyLoss()

    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data = resize(data)
        data, target = data.cuda(args.rank), target.cuda(args.rank)

        for param in model.parameters():
            param.grad = None

        logits = model(data)
        # if(args.name == 'MedSAM' or args.name == 'LVM_Vit'):
        #     logits = logits.unsqueeze(0) 
        loss = loss_func(logits, target)
        loss.backward()
        
        optimizer.step()
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


def val_epoch(model, loader, epoch, args):
    model.eval()
    start_time = time.time()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["image"], batch_data["label"]

            data = resize(data)
            data, target = data.cuda(args.rank), target.cuda(args.rank)

            with autocast(enabled=args.amp):
                logits = model(data)
                # if(args.name == 'MedSAM' or args.name == 'LVM_Vit'):
                #     logits = logits.unsqueeze(0) 

            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)

            # 保证 target 是一维类别id
            target_np = target.cpu().numpy()
            if target_np.ndim > 1:
                target_np = np.argmax(target_np, axis=1)
            all_targets.extend(target_np)

            if args.rank == 0:
                print(
                    "Val {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                    "time {:.2f}s".format(time.time() - start_time),
                )
    all_probs = np.vstack(all_probs)
    all_targets = np.array(all_targets).astype(int)
    n_classes = all_probs.shape[1]
    unique_classes = np.unique(all_targets)

    # 若类别不全，单独计算实际出现类别的AUC
    auc_scores = []
    for class_idx in unique_classes:
        binary_targets = (all_targets == class_idx).astype(int)
        try:
            auc = metrics.roc_auc_score(binary_targets, all_probs[:, class_idx])
            print(f"类别 {class_idx} AUC: {auc:.4f}")
            auc_scores.append(auc)
        except Exception as e:
            print(f"类别 {class_idx} AUC计算失败: {e}")

    if len(unique_classes) == n_classes:
        # 验证集包含所有类别，可用官方多分类AUC
        try:
            auc_macro = metrics.roc_auc_score(all_targets, all_probs, multi_class='ovr', average='macro')
            print(f"多分类AUC (macro-ovr): {auc_macro:.4f}")
            return auc_macro
        except Exception as e:
            print(f"多分类AUC计算失败: {e}")

    if auc_scores:
        mean_auc = np.mean(auc_scores)
        print(f"实际出现类别的平均AUC: {mean_auc:.4f}")
        return mean_auc
    else:
        print("没有有效的AUC可以计算")
        return 0.0


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
    args,
    scheduler=None,
    start_epoch=0,
):
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
            model, train_loader, optimizer, scheduler, scaler=scaler, epoch=epoch, args=args
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
                args=args,
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

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
