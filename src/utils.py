import math
import torch
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


def psnr(a, b, data_range=1.0):
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return compare_psnr(a, b, data_range=data_range)


def ssim(a, b, data_range=1.0):
    return compare_ssim(a, b, data_range=data_range)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        bestf = filename.replace('.pth.tar', '_best.pth.tar')
        torch.save(state, bestf)
