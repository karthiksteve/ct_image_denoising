# Model Checkpoints

## Trained Models
This folder should contain trained model weights.

### Expected Files
- `best_model.pth.tar` - Best performing model checkpoint (based on validation PSNR)

### Current Status
⚠️ **No trained model checkpoint found**

To generate the model:
1. Navigate to `../src/`
2. Run training: `python train.py --config configs/config.yaml`
3. Best model will be saved automatically to `checkpoints/best.pth.tar`
4. Copy it here for GitHub submission

### Loading Trained Model
```python
import torch
from src.models.networks import GeneratorUNetDual

# Load model
G = GeneratorUNetDual(in_channels=1, base_features=32)
checkpoint = torch.load('models/best_model.pth.tar', map_location='cpu')
G.load_state_dict(checkpoint['G_state'])
G.eval()

# Use for inference
# denoised, noise_residual = G(noisy_input)
```

## File Format
- PyTorch checkpoint (`.pth.tar` or `.pt`)
- Contains: Generator state_dict, epoch, metrics, optimizer states
