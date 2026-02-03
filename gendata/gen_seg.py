import os
from os.path import join
from time import time
import argparse
import SimpleITK as sitk
import torch
from tqdm import trange

from utils.dataloader_gen_online import DatasetGEN3D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic NIfTI dataset using DatasetGEN3D")
    parser.add_argument("--data_amount", type=int, default=1000, help="number of samples to generate")
    parser.add_argument("--root", type=str, default="/home/yweibs/scratch/yweibs/data/", help="root folder to save data")
    parser.add_argument("--img_dir", type=str, default="Image", help="subfolder for images")
    parser.add_argument("--lab_dir", type=str, default="Label", help="subfolder for labels")
    parser.add_argument("--shape", type=int, default=128, help="spatial size used by generator (e.g. 128)")
    parser.add_argument("--num_label", type=int, default=None, help="number of labels (default: shape//16)")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="device for generation")
    parser.add_argument("--seed", type=int, default=None, help="optional seed for reproducibility")
    parser.add_argument("--dry_run", action="store_true", help="Generate data but do not write files to disk (useful for testing)")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    num_label = args.num_label if args.num_label is not None else args.shape // 16

    gen = DatasetGEN3D(in_shape=(args.shape,), num_label=num_label)

    root = args.root
    img_dir = args.img_dir
    Lab_dir = args.lab_dir

    os.makedirs(join(root, img_dir), exist_ok=True)
    os.makedirs(join(root, Lab_dir), exist_ok=True)

    start_time = time()

    it = trange(args.data_amount, desc="Generating samples")
    for i in it:
        name = f"{int(time() * 1e6)}_{i}"
        img, lab = gen.__getitem__()
        lab_arg = torch.argmax(lab, dim=1)
        lab_np = lab_arg.data.cpu().numpy()[0].astype(np.int8)
        img_np = img.data.cpu().numpy()[0, 0].astype(np.float32)

        if not args.dry_run:
            img_sitk = sitk.GetImageFromArray(img_np)
            lab_sitk = sitk.GetImageFromArray(lab_np)
            sitk.WriteImage(img_sitk, join(root, img_dir, name + '.nii'))
            sitk.WriteImage(lab_sitk, join(root, Lab_dir, name + '.nii'))
        it.set_postfix(idx=i)

    synthesis_time = time() - start_time
    print(f"Synthesis time: {synthesis_time:.2f} seconds | samples generated: {args.data_amount} | saved to: {root}")


if __name__ == '__main__':
    main()