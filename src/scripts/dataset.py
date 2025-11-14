import os
import glob
import numpy as np
from torch.utils.data import Dataset
import torch

class CTPairDataset(Dataset):
    """Loads .npz files saved by preprocess.py containing 'noisy' and 'clean'."""
    def __init__(self, root_dir, transform=None):
        self.files = sorted(glob.glob(os.path.join(root_dir, '*.npz')))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        f = self.files[idx]
        data = np.load(f)
        noisy = data['noisy'].astype(np.float32)
        clean = data['clean'].astype(np.float32)
        # add channel dim
        noisy = noisy[np.newaxis, ...]
        clean = clean[np.newaxis, ...]
        noisy = torch.from_numpy(noisy)
        clean = torch.from_numpy(clean)
        if self.transform:
            noisy, clean = self.transform(noisy, clean)
        return noisy, clean
