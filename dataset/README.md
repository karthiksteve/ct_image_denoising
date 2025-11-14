# Dataset

## Contents
- **pairs/**: Noisy-clean CT image pairs stored as NPZ files (1,040 pairs)
- **sample_images/**: Sample ground truth clean CT images (50 PNG samples)

## Data Source
CT scan slices extracted from HDF5 format medical imaging datasets.

## Preprocessing
- Noise simulation: Gaussian noise (σ=0.05) added to clean images
- Format: NumPy compressed arrays (.npz) with keys: `noisy`, `clean`
- Resolution: Variable (medical CT standard)

## Usage
```python
import numpy as np
data = np.load('pairs/slice_0000.npz')
noisy = data['noisy']  # Noisy input
clean = data['clean']  # Ground truth
```

## Note
Full dataset (3,684 images) available separately due to size constraints.
This repository includes a representative subset for demonstration.
