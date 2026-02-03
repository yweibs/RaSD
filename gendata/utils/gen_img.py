import os
import random

import time
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as nnf
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter

from utils.gen_lab import draw_perlin, transform
# from Transform import SpatialTransform


def minmax_norm(x, axis=None):
    x_min = torch.min(x)
    x_max = torch.max(x)
    x = x - x_min
    y = x_max - x_min
    return x/(y+1e-6)

def subsample_axis(x, stride_min=1, stride_max=8, axes=None, prob=1, upsample=True):
    # Validate axes.
    num_dim = len(x.shape)
    if axes is None:
        axes = range(num_dim)
    if np.isscalar(axes):
        axes = [axes]
    assert all(i in range(num_dim) for i in axes), 'invalid axis passed'

    # Draw axis and thickness.
    assert 0 < stride_min and stride_min <= stride_max, 'invalid strides'
    ind = torch.rand(size=[])*len(axes)
    ax = axes[ind]
    width = x.shape[ax]
    thick = torch.rand(size=[])*(stride_max-stride_min) + stride_min

    # Decide whether to downsample.
    assert 0 <= prob <= 1, f'{prob} not a probability'
    if prob < 1:
        rand_bit = torch.less(torch.rand(size=[]), prob)
        rand_not = torch.logical_not(rand_bit)
        thick = thick * rand_bit.type(thick.dtype) + rand_not.type(thick.dtype)

    # Resample.
    num_slice = width.dype(thick.dtype) / thick + 0.5
    num_slice = num_slice.type(width.dtype)
    ind = torch.linspace(start=0, end=width - 1, step=num_slice)
    ind = (ind + 0.5).type(width.dtype)
    x = torch.take(x, ind, axis=ax)
    if upsample:
        ind = torch.linspace(start=0, end=np.shape(x)[ax] - 1, step=width)
        ind = (ind + 0.5).type(width.dtype)
        x = torch.take(x, ind, axis=ax)

    return x

def Subsample(x, stride_min=1, stride_max=8, axes=None, prob=1, upsample=True):

    ndims = len(x.shape) - 2
    assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'

    allowed = range(1, ndims + 1)
    axes = normalize_axes(axes, x.shape, allowed, none_means_all=True)

    if prob == 0 or stride_max == 1:
        return x

    shape = x.shape
    x = subsample_axis(x, stride_min=stride_min, stride_max=stride_max, axes=axes, prob=prob, upsample=upsample)

    # Avoid unnecessary dynamic shapes (showing up as `None`).
    return torch.reshape(x, shape) if upsample else x



def GaussianBlur(x, sigma=None, level=None, random=False, min_sigma=0, isotropic=False, device='cpu'):
    n_dims = len(x.shape[2:])
    if level is not None:
        sigma = (level - 1) ** 2

    if isotropic and not random:
        raise ValueError('For non-random blurring, isotropy is implicitly controlled by the '
                         'number of sigmas provided. Set `isotropic` only for random blur.')

    def _normalize_sigma(sigma, ndims):
        sigma = np.ravel(sigma)
        sigma = sigma.tolist()
        if len(sigma) not in (1, ndims):
            raise ValueError(f'1 or {ndims} sigmas expected in {ndims}D space, got {len(sigma)}')

        if any(s < 0 for s in sigma):
            raise ValueError('Gaussian blur sigma must not be less than 0')

        if len(sigma) > 1 and isotropic:
            raise ValueError(f'random isotropic blur requires a single sigma, got {len(sigma)}')

        if len(sigma) == 1:
            sigma = sigma * ndims
        return sigma

    sigma = _normalize_sigma(sigma, n_dims)
    min_sigma = _normalize_sigma(min_sigma, n_dims)

    if isotropic and random:
        sigma = sigma[:1]
        min_sigma = min_sigma[:1]

    if not any(s > 0 for s in sigma):
        return x

    kernel = gaussian_kernel(sigma=sigma, random=random, min_sigma=min_sigma, separate=True, dtype=np.float32)

    ones = np.ones(n_dims, np.int32)
    zeros = np.zeros(n_dims, np.int32)
    for i in range(n_dims):
        # print(kernel[i].shape)
        k = kernel[i].reshape((1, 1, *ones[:i], -1, *ones[i+1:])).astype(np.float32)
        k = torch.from_numpy(k).to(device)
        x = nnf.conv3d(x, k, padding=(*zeros[:i], k.shape[i+2]//2, *zeros[i+1:]), stride=1, dilation=1)

    return x


def GaussianNoise(x, noise_min=0.01, noise_max=0.10, noise_only=False, absolute=False, axes=(0, -1), device='cpu'):

    num_dim = len(x.shape[2:])
    axes = np.ravel(axes)
    axes = [ax + num_dim if ax < 0 else ax for ax in axes]
    assert all(0 <= ax < num_dim for ax in axes), 'invalid axes'


    if noise_max == 0 and not noise_only:
        return x

    # Shapes.
    shape_out = x.shape
    # shape_sd = []
    # for i, _ in enumerate(x.shape):
    #     shape_sd.append(shape_out[i] if i in axes else 1)

    # Standard deviation.
    sd = random.uniform(noise_min, noise_max)
    if not absolute:
        sd *= torch.max(torch.abs(x))

    noise = torch.normal(size=shape_out, mean=0, std=sd, device=device)
    noise = noise + x

    return noise

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

def random_blur_rescale(x, std_min=8 / 2.355, std_max=32 / 2.355, isotropic=False, reduce=torch.std, device='cpu'):
    n_dim = len(x.shape[2:])

    rand = np.random.default_rng() 
    seeds = rand.integers(np.iinfo(int).max, size=n_dim) 

    kernel = [gaussian_kernel(sigma=std_max, separate=True, random=True, min_sigma=std_min, seed=s).astype(np.float32) for s in seeds] # 高斯核
    if isotropic: 
        kernel = kernel[:1] * n_dim

    # Rescaling.
    before = reduce(x) 
    ones = np.ones(n_dim, np.int32)
    zeros = np.zeros(n_dim, np.int32)
    for i in range(n_dim): 
        # print(kernel[i].shape)
        k = kernel[i].reshape((1, 1, *ones[:i], -1, *ones[i+1:]))
        k = torch.from_numpy(k).to(device)
        x = nnf.conv3d(x, k, padding=(*zeros[:i], k.shape[i+2]//2, *zeros[i+1:]), stride=1, dilation=1)

    after = reduce(x) 
    return x * before/(after+1e-6) 


def normalize_axes(axes, shape, allowed=None, none_means_all=False):
    ndims = len(shape)
    if allowed is None:
        allowed = range(ndims)
    if np.isscalar(allowed):
        allowed = [allowed]
    assert all(ax in range(ndims) for ax in allowed), f'allowed axes {allowed} out of bounds'

    # Axis inputs.
    if axes is None:
        axes = allowed if none_means_all else []
    if np.isscalar(axes):
        axes = [axes]

    # Convert negative indices.
    orig = axes
    axes = [ax + ndims if ax < 0 else ax for ax in axes]

    # Validate.
    for ax, inp in zip(axes, orig):
        if ax not in allowed:
            raise IndexError(f'axis {inp} outside {allowed}')

    # Sort, remove duplicates.
    return tuple(set(axes))

def draw_perlin_full(shape,
                     noise_min=0.01,
                     noise_max=1,
                     fwhm_min=4,
                     fwhm_max=32,
                     isotropic=False,
                     reduce=torch.std,
                     axes=None,
                     dtype=torch.float32,
                     device='cpu'):

    # Dimensions. Increment axes if we prepend a batch dimension.
    # axes = normalize_axes(axes, shape, none_means_all=False)

    # SD shape. Index into rather than iterate over tensor.
    # shape_sd = [shape[i] if i in axes else 1 for i in range(len(shape))]
    # print(shape_sd)

    if not hasattr(fwhm_min, '__iter__'):
        fwhm_min = [fwhm_min]
    if not hasattr(fwhm_max, '__iter__'):
        fwhm_max = [fwhm_max]
    assert len(fwhm_min) == len(fwhm_max), 'different number of lower and upper bounds'

    # Levels.
    out = []
    for low, upp in zip(fwhm_min, fwhm_max):
        noise = random.uniform(noise_min, noise_max)
        # torch.rand(size=shape_sd, device=device, dtype=dtype)*(noise_max - noise_min) + noise_min
        noise = torch.normal(size=shape, mean=0., std=noise, dtype=dtype, device=device)
        noise = random_blur_rescale(
            noise,
            std_min=low / 2.355,
            std_max=upp / 2.355,
            isotropic=isotropic,
            reduce=reduce,
            device=device
        ) # low, upp 是 FWHM
        out.append(noise)

    # Output. Compute mean to maintain the noise level when adding scales.
    out = torch.cat(out, dim=0)
    out = torch.mean(out, dim=0)
    return out


def PerlinNoise(shape, noise_min=0.01, noise_max=1, fwhm_min=4, fwhm_max=32, isotropic=False, reduce=torch.std, axes=None, device='cpu'):

    axes = normalize_axes(axes, shape, none_means_all=False)
    
    noise = draw_perlin_full(shape,
                            noise_min=noise_min,
                            noise_max=noise_max,
                            isotropic=isotropic,
                            fwhm_min=fwhm_min,
                            fwhm_max=fwhm_max,
                            axes=[ax - 1 for ax in axes],
                            reduce=reduce,
                             device=device)

    return noise

def labels_to_image(
        labels,
        num_label=16,
        num_chan=1,
        mean_min=0,
        mean_max=1,
        noise_min=0.05,
        noise_max=0.4,
        zero_background=0,
        blur_min=0,
        blur_max=1,
        bias_min=0.01,
        bias_max=0.1,
        bias_blur_min=16,
        bias_blur_max=32,
        bias_func=torch.exp,
        slice_stride_min=1,
        slice_stride_max=8,
        slice_prob=0,
        slice_axes=None,
        normalize=True,
        gamma=0.5,
        device='cpu',
        ):
    labels_in = range(num_label)
    integer_type = torch.long

    # Dimensions.
    num_dim = len(labels.shape[2:])
    batch_size = labels.shape[0]

    # Generation labels. Default to all input labels.
    if not isinstance(labels_in, dict):
        labels_in = {i: i for i in labels_in}
    labels_gen = set(labels_in.values())

    # Rebase into [0, N): LUT from input to generation label to index.
    ind = {gen: i for i, gen in enumerate(labels_gen)}
    lut = [ind.get(labels_in.get(i), 0) for i in range(max(labels_in) + 1)]
    lut = torch.from_numpy(np.asarray(lut)).type(integer_type).to(device)

    labels = labels.type(integer_type)
    indices = lut[labels]


    # Intensity means and standard deviations.
    mean = torch.rand(size=(batch_size, num_chan, num_label), device=device)*(mean_max-mean_min) + mean_min

    # Label intensities.
    off_chan = torch.arange(num_chan, device=device) * num_label
    off_batch = torch.arange(batch_size, device=device) * num_chan * num_label
    indices += torch.reshape(off_batch, shape=(-1, 1, *[1] * num_dim)) + off_chan
    mean = torch.reshape(mean, shape=(-1,))[indices]
    image = mean
    
    # Bias field.
    if bias_max > 0:
        bias_field = PerlinNoise(image.shape, noise_min=bias_min, noise_max=bias_max, isotropic=False,
            fwhm_min=bias_blur_min, fwhm_max=bias_blur_max, reduce=torch.max, device=device)
        bias_field = bias_func(bias_field)
        image = image * bias_field

    # Noise.
    image = GaussianNoise(image, noise_min, noise_max, device=device)

    # image_2_np = image.data.cpu().numpy()[0,0,128,:,:]  # 获取形状为 (256, 256)
    # image_2_np = ((image_2_np - image_2_np.min()) / (image_2_np.max() - image_2_np.min()) * 255).astype(np.uint8)
    # Image.fromarray(image_2_np, mode='L').save('test_img_2.png')

    # Background clearing.
    if zero_background > 0:
        bg_rand = torch.rand(size=(batch_size, *[1] * num_dim, 1))
        bg_zero = torch.less(bg_rand, zero_background)
        bg_zero = torch.logical_and(np.equal(labels, 0), bg_zero)
        bg_zero = torch.logical_xor(True, bg_zero)
        image *= bg_zero
    
    # Blur.
    image = GaussianBlur(image, sigma=blur_max, min_sigma=blur_min, random=True, device=device)
    # slice spacing
    image = Subsample(image, prob=slice_prob, stride_min=max(1, slice_stride_min), stride_max=max(1, slice_stride_max), axes=slice_axes)

    # Intensity manipulations.
    if normalize:
        image = minmax_norm(image)
    if gamma > 0:
        assert 0 < gamma < 1, f'gamma value {gamma} outside interval [0, 1)'
        gamma = torch.rand(size=(batch_size, *[1] * num_dim, num_chan), device=device)*(2*gamma)+(1-gamma)
        image = torch.pow(image, gamma)

    return image

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    in_shape = (256,) * 3
    num_dim = len(in_shape)
    num_label = 16
    num_maps = 40

    # start_time = time.time()  
    
    lab = draw_perlin(out_shape=(num_label, 1, *in_shape), scales=(32, 64, 128), max_std=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    flow = draw_perlin(out_shape=(num_label, num_dim, *in_shape), scales=(16, 32, 64, 128), max_std=16, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    warpped_lab = transform(lab, flow)
    warpped_lab = torch.argmax(warpped_lab, dim=0, keepdim=True).type(torch.float32)

    image = labels_to_image(warpped_lab, num_label, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # end_time = time.time() 
    # elapsed_time = end_time - start_time  
    # print(f"Elapsed time: {elapsed_time:.2f} seconds") 