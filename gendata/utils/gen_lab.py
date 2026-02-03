import os
import random

import numpy as np
import torch
import torch.nn.functional as nnf
import SimpleITK as sitk
import random
from PIL import Image
import os


def gaussian_kernel(sigma, windowsize=None, indexing='ij', separate=False, random=False, min_sigma=0, dtype=np.float32,
                    seed=None):
    # assert dtype.is_floating, f'{dtype.name} is not a real floating-point type'

    # Kernel width.
    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]
    if not isinstance(min_sigma, (list, tuple)):
        min_sigma = [min_sigma] * len(sigma)
    sigma = [max(f, np.finfo(dtype).eps) for f in sigma]
    min_sigma = [max(f, np.finfo(dtype).eps) for f in min_sigma]

    # Kernel size.
    if windowsize is None:
        windowsize = [np.round(f * 3) * 2 + 1 for f in sigma]
    if not isinstance(windowsize, (list, tuple)):
        windowsize = [windowsize]
    if len(sigma) != len(windowsize):
        raise ValueError(f'sigma {sigma} and width {windowsize} differ in length')

    # Precompute grid.
    center = [(w - 1) / 2 for w in windowsize]
    mesh = [np.arange(w) - c for w, c in zip(windowsize, center)]
    mesh = [-0.5 * x**2 for x in mesh]
    if not separate:
        mesh = np.meshgrid(*mesh, indexing=indexing)
    mesh = [np.array(m) for m in mesh]

    # Exponents.
    if random:
        seeds = np.random.default_rng(seed).integers(np.iinfo(int).max, size=len(sigma))
        max_sigma = sigma
        sigma = []
        for a, b, s in zip(min_sigma, max_sigma, seeds):
            sigma.append(np.random.uniform(size=(1,), low=a, high=b))
    exponent = [m / s**2 for m, s in zip(mesh, sigma)]

    # Kernel.
    if not separate:
        exponent = [np.sum(np.stack(exponent), axis=0)]
    kernel = [np.exp(x) for x in exponent]
    kernel = [x / np.sum(x) for x in kernel]

    return kernel if len(kernel) > 1 else kernel[0]

def draw_perlin(out_shape, scales, min_std=0, max_std=1, dtype=torch.float32, device='cpu'):
    out = torch.zeros(size=out_shape, dtype=dtype, device=device)
    for scale in scales:
        sample_shape = (np.asarray(out_shape[2:]) // scale).tolist()
        sample_shape = [out_shape[0], out_shape[1], *sample_shape]

        std = random.uniform(min_std, max_std)
        gauss = torch.normal(size=sample_shape, mean=0, std=std, device=device)

        if scale != 1:
            gauss = resize(gauss, out_shape[2:])
            
        out += gauss
    return out

def save_noise_frames(noise_frames, base_path, scales):
    for i, frame in enumerate(noise_frames):
        filename = f'perlin_noise_scale_{scales[i] if i < len(scales) else "final"}.png'
        filepath = os.path.join(base_path, filename)
        frame = frame.squeeze()  
        frame = (frame - frame.min()) / (frame.max() - frame.min()) 
        frame = frame.cpu().numpy() 
        pil_img = Image.fromarray((frame * 255).astype(np.uint8))  
        pil_img.save(filepath) 

def draw_perlin_flow(out_shape, scales, min_std=0, max_std=1, min_alp=0, max_alp=100, dtype=torch.float32, device='cpu'):
    n_dim = len(out_shape[2:])
    # print(scales)
    out = torch.zeros(size=out_shape, dtype=dtype, device=device)

    ones = np.ones(n_dim, np.int32)
    zeros = np.zeros(n_dim, np.int32)

    rand = np.random.default_rng()
    seeds = rand.integers(np.iinfo(int).max, size=n_dim)

    for scale in scales:
        sample_shape = (np.asarray(out_shape[2:]) // scale).tolist()
        sample_shape = [out_shape[0], out_shape[1], *sample_shape]

        gauss = torch.rand(size=sample_shape, device=device)
        kernel = [
            gaussian_kernel(sigma=max_std, separate=True, random=True, min_sigma=min_std, seed=s).astype(np.float32) for
            s in seeds]

        alp = random.uniform(min_alp, max_alp)
        for i in range(n_dim):
            # print(kernel[i].shape)
            k = kernel[i].reshape((1, 1, *ones[:i], -1, *ones[i + 1:]))
            k = torch.from_numpy(k).to(device)
            k = torch.cat((k,) * n_dim, dim=0)
            gauss = nnf.conv3d(gauss, k, padding=(*zeros[:i], k.shape[i + 2] // 2, *zeros[i + 1:]), stride=1, dilation=1, groups=n_dim)*alp

        # gauss = nnf.conv3d(gauss, kernal, padding=1, groups=out_shape[1])/sum((3, ) * len(out_shape[2:]))

        out += gauss if scale == 1 else resize(gauss, out_shape[2:])

    return out

def resize(vol, out_shape, mode="bilinear"):
    vol_shape = vol.shape[2:]
    ndims = len(vol_shape)
    if ndims == 2:
        mode = 'bilinear'
    elif ndims == 3:
        mode = 'trilinear'
    else:
        assert "dim err"
    vol = torch.nn.Upsample(size=out_shape, mode=mode, align_corners=True)(vol)
    return vol

def transform(src, flow, mode="bilinear"):
    shape = flow.shape[2:]

    vectors = [torch.arange(0, s) for s in shape]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)  # y, x, z
    grid = torch.unsqueeze(grid, 0)  # add batch
    grid = grid.type(flow.dtype)
    grid = grid.cuda()

    new_locs = grid + flow

    for i in range(len(shape)):
        new_locs[:, i, ...] = 2*(new_locs[:,i,...]/(shape[i]-1) - 0.5)

    if len(shape) == 2:
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1,0]]
    elif len(shape) == 3:
        new_locs = new_locs.permute(0, 2, 3, 4, 1)
        new_locs = new_locs[..., [2,1,0]]
    src = nnf.grid_sample(src, new_locs, mode=mode)
    return src

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    in_shape = [256,] * 3
    # batch_size = 1
    num_dim = len(in_shape)
    num_label = 16
    num_maps = 40

    out = draw_perlin(out_shape=[num_label, 1, *in_shape], scales=[32, 64, 128], max_std=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    out_2 = draw_perlin(out_shape=[num_label, num_dim, *in_shape], scales=[32, 64, 128], max_std=16, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    base_path = './img'
    
    warpped_im = transform(out, out_2)
    # print(warpped_im.shape)
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    save_noise_frames(warpped_im, base_path, scales=[1])
