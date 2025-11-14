"""
Preprocessing helper:
- Inputs: place grayscale CT slices (PNG/JPG/NPY) under data/raw/normal/
- Outputs: paired arrays saved to data/processed/ as .npz with keys 'noisy' and 'clean'

Two simulation modes:
- simple: add Gaussian + Poisson-like noise (fast, for demo)
- radon: simulate sinogram Poisson noise and reconstruct with iradon (slower, more realistic)

Run: python scripts/preprocess.py --mode simple --raw_dir data/raw/normal --out_dir data/processed
"""
import os
import argparse
import numpy as np
from skimage import io, img_as_float
from skimage.transform import radon, iradon
import glob


def simulate_simple_noise(img, noise_level=0.05):
    # img assumed float in [0,1]
    noisy = img + np.random.normal(0, noise_level, img.shape)
    noisy = np.clip(noisy, 0.0, 1.0)
    return noisy


def simulate_radon_poisson(img, count_level=1e5):
    # img in [0,1] -> attenuation: small hack for demo
    # compute radon, simulate Poisson on photon counts, reconstruct
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sino = radon(img, theta=theta, circle=True)
    # convert to counts (I0 * exp(-sino))
    I0 = count_level
    counts = np.random.poisson(I0 * np.exp(-sino))
    counts = np.maximum(counts, 1)
    sino_noisy = -np.log(counts / float(I0))
    rec = iradon(sino_noisy, theta=theta, circle=True, filter_name='ramp')
    rec = rec - rec.min()
    rec = rec / (rec.max() + 1e-8)
    rec = np.clip(rec, 0.0, 1.0)
    return rec


def process_folder(raw_dir, out_dir, mode='simple', noise_level=0.05, compress=True, skip_existing=False):
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(raw_dir, "*.*")))
    if not files:
        print('No files found in', raw_dir)
        return
    for i, f in enumerate(files):
        out_path = os.path.join(out_dir, f'slice_{i:04d}.npz')
        if skip_existing and os.path.exists(out_path):
            if i % 200 == 0:
                print('skip existing up to', i)
            continue
        try:
            img = img_as_float(io.imread(f, as_gray=True))
        except Exception as e:
            print('skip', f, e); continue
        if mode == 'simple':
            noisy = simulate_simple_noise(img, noise_level=noise_level)
        else:
            noisy = simulate_radon_poisson(img)
        if compress:
            np.savez_compressed(out_path, noisy=noisy.astype(np.float32), clean=img.astype(np.float32))
        else:
            np.savez(out_path, noisy=noisy.astype(np.float32), clean=img.astype(np.float32))
        if i % 50 == 0:
            print('processed', i)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--raw_dir', default='data/raw/normal')
    p.add_argument('--out_dir', default='data/processed')
    p.add_argument('--mode', choices=['simple','radon'], default='simple')
    p.add_argument('--noise_level', type=float, default=0.05)
    p.add_argument('--no_compress', action='store_true', help='Save NPZ without compression (faster)')
    p.add_argument('--skip_existing', action='store_true', help='Skip writing files that already exist (resume)')
    args = p.parse_args()
    process_folder(args.raw_dir, args.out_dir, mode=args.mode, noise_level=args.noise_level, compress=not args.no_compress, skip_existing=args.skip_existing)
