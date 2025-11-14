# Noise Aware Content-Noise Complementary GAN (Demo)

This repository is a demo scaffold for "Noise Aware Content-Noise Complementary GAN with Local and Global Discrimination for Low-Dose CT Denoising".

Contents:
- `scripts/` : preprocessing, dataset, and utilities
- `models/`  : model definitions (generator + local/global discriminators)
- `train.py` : training loop
- `evaluate.py` : evaluation and image output
- `data/` : data/raw for originals and data/processed for preprocessed pairs

Quick start (PowerShell):
1. Create a venv and install requirements
   python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
2. Prepare data (see `scripts/preprocess.py`): place CT slices into `data/raw/normal/` then run preprocessing to generate low-dose/normal pairs
3. Train: `python train.py --config configs/config.yaml`
4. Evaluate: `python evaluate.py --config configs/config.yaml`

See `docs/` for more details.
