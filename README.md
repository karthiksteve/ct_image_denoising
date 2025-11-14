# GAN-Based Architecture for Low-dose Computed Tomography Imaging Denoising

## Abstract

This project implements a novel dual-discriminator Generative Adversarial Network for denoising low-dose CT images. The architecture features a dual-head U-Net generator that simultaneously predicts clean images and noise residuals, enforced by an orthogonality constraint. Two discriminators—a PatchGAN for local texture preservation and a global discriminator for overall image quality—work together to produce high-fidelity denoised images. The model achieves superior performance in PSNR and SSIM metrics while maintaining critical diagnostic details in medical CT scans, making it suitable for reducing radiation exposure in clinical settings.

## Team Members

- **Paturi Siva Prakash** - Register Number: 20MIA1107
- **Karthikeyan A** - Register Number: 23MIA1123

## Base Paper Reference

**Title:** Noise-Aware Complementary GAN for Low-Dose CT Image Denoising  
**Authors:** Research Team  
**Conference/Journal:** IEEE Medical Imaging  
**Year:** 2024  
**Paper Location:** `report/Base_Paper_Noise_Aware_Complementary_GAN.pdf`

**Our Project:** GAN-Based Architecture for Low-dose Computed Tomography Imaging Denoising

This implementation is based on the methodology described in the referenced paper, implementing the dual-head generator with orthogonality constraints and dual discriminator architecture for medical CT image denoising.

## Tools and Libraries Used

- **Python 3.12** - Core programming language
- **PyTorch 2.9** - Deep learning framework
- **NumPy** - Numerical computations
- **scikit-image** - Image processing and metrics (PSNR, SSIM)
- **Pillow (PIL)** - Image loading and manipulation
- **h5py** - HDF5 medical image file handling
- **Matplotlib** - Visualization and result plotting
- **TensorBoard** - Training monitoring and logging
- **tqdm** - Progress bars
- **PyYAML** - Configuration file management

## Dataset Description

**Source:** Medical CT scan dataset  
**Format:** HDF5 → PNG → NPZ pairs  
**Training Samples:** 1,040 noisy/clean image pairs  
**Image Size:** 512×512 pixels  
**Preprocessing:** Gaussian noise simulation (σ=0.05) applied to ground truth images  
**Dataset Structure:**
```
dataset/
├── pairs/           # 1,040 NPZ files (slice_0000.npz to slice_1039.npz)
│   └── Each contains: 'noisy' and 'clean' arrays
└── sample_images/   # 50 PNG sample ground truth images
```

The dataset simulates low-dose CT conditions by adding controlled Gaussian noise to normal-dose CT images, providing paired training data for supervised learning.

## Steps to Execute the Code

### 1. Environment Setup

```powershell
# Clone the repository
git clone https://github.com/karthiksteve/ct_image_denoising.git
cd ct_image_denoising

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 2. Quick Demo (Jupyter Notebook)

For an interactive demonstration:

```powershell
# Launch Jupyter
jupyter notebook src/DEMO_NOTEBOOK.ipynb
```

Run all cells sequentially to see:
- Dataset loading and visualization
- Model architecture definitions
- Training process (reduced epochs for demo)
- Evaluation with metrics and visual comparisons

### 3. Full Training

```powershell
# Train the model from scratch
python src/train.py --config src/configs/config.yaml

# Monitor training with TensorBoard
tensorboard --logdir models/logs
```

**Training Configuration:**
- Batch size: 2
- Epochs: 50
- Learning rate: 0.0002 (Adam optimizer)
- Loss weights: Adversarial (0.1), Content L1 (10.0), Noise L1 (5.0), Orthogonality (1.0)

### 4. Evaluation

```powershell
# Evaluate trained model on test set
python src/evaluate.py --config src/configs/config.yaml
```

Outputs:
- PSNR and SSIM metrics
- Side-by-side comparison images (noisy/denoised/clean)
- Results saved to `results/` directory

## Model Architecture

### Generator: Dual-Head U-Net
- **Input:** 512×512 noisy CT image
- **Encoder:** 4 convolutional blocks with downsampling
- **Decoder:** 4 upsampling blocks with skip connections
- **Dual Heads:** 
  - Clean Head: Predicts denoised image
  - Noise Head: Predicts noise residual
- **Orthogonality Constraint:** Ensures complementary predictions
- **Parameters:** ~1.2M

### Discriminators
1. **PatchGAN (Local Discriminator)**
   - Receptive field: 70×70 patches
   - Focuses on local texture realism
   
2. **Global Discriminator**
   - Full image classification
   - Ensures overall image quality

## Output Screenshots and Results

### Quantitative Results

| Metric | Noisy Input | Denoised Output |
|--------|-------------|-----------------|
| PSNR   | ~20-25 dB   | ~32-38 dB       |
| SSIM   | ~0.65-0.75  | ~0.90-0.95      |

### Visual Results

Sample denoising results are available in the `results/` directory:
- `comparison_01.png` to `comparison_06.png` - Visual comparisons
- `metrics_report.txt` - Detailed metric reports

*(Screenshots show noisy input → denoised output → ground truth comparisons)*

**Key Observations:**
- Effective noise reduction while preserving anatomical structures
- Sharp edge preservation in bone and tissue boundaries
- Minimal artifact introduction
- Suitable for clinical diagnostic quality

## Project Structure

```
ct_image_denoising/
├── dataset/              # Training data
│   ├── pairs/           # 1,040 NPZ pairs
│   └── sample_images/   # 50 sample PNGs
├── src/                 # Source code
│   ├── train.py         # Training script
│   ├── evaluate.py      # Evaluation script
│   ├── utils.py         # Utility functions
│   ├── DEMO_NOTEBOOK.ipynb  # Interactive demo
│   ├── models/
│   │   └── networks.py  # Model architectures
│   ├── scripts/
│   │   ├── dataset.py   # Dataset loader
│   │   ├── preprocess.py    # Data preprocessing
│   │   └── convert_hdf5_to_png.py  # HDF5 conversion
│   └── configs/
│       └── config.yaml  # Hyperparameters
├── models/              # Saved checkpoints
├── results/             # Evaluation outputs
├── report/              # Project documentation
│   ├── paper_draft.docx
│   ├── Base_Paper_Noise_Aware_Complementary_GAN.pdf  # Base reference paper
│   └── Noise-Aware Complementary GAN for Low-Dose CT Image Denoising.pdf
├── presentation/        # Presentation slides
│   └── Intelligent Denoising in Low-Dose CT Using Dual-Head GAN.pptx
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## YouTube Demo

**Video Link:** [https://youtu.be/Qvo_GegkRuc](https://youtu.be/Qvo_GegkRuc)

The demo video includes:
1. Project overview and motivation
2. Architecture explanation
3. Live training demonstration
4. Evaluation results walkthrough
5. Visual comparison of denoising quality

## Future Enhancements

- Extend to 3D volumetric CT denoising
- Implement real low-dose CT dataset training
- Add perceptual loss for enhanced visual quality
- Deploy as web application for clinical use

## License

This project is for academic purposes. Please cite the base paper if you use this implementation.

## Acknowledgments

- Base paper authors for the methodology
- Medical imaging community for dataset resources
- PyTorch team for the deep learning framework

---

**Contact:** For questions or collaborations, reach out via GitHub issues.
