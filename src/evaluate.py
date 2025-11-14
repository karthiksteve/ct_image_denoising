import argparse
import yaml
import os
import torch
from torch.utils.data import DataLoader
from scripts.dataset import CTPairDataset
from models.networks import GeneratorUNetDual
from utils import psnr, ssim


def load_config(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate(config, checkpoint=None):
    # Device selection: Intel Arc (xpu), CUDA, or CPU
    device_str = config['training']['device']
    if device_str == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"Using Intel Arc GPU: {torch.xpu.get_device_name(0)}")
    elif device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    ds = CTPairDataset(config['data']['pairs_dir'])
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    G = GeneratorUNetDual(in_channels=1, base_features=config['model']['base_features']).to(device)
    if checkpoint:
        ck = torch.load(checkpoint, map_location=device)
        G.load_state_dict(ck['G_state'])
    G.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    n = 0
    os.makedirs('outputs', exist_ok=True)
    with torch.no_grad():
        for i, (noisy, clean) in enumerate(loader):
            noisy = noisy.to(device); clean = clean.to(device)
            pred_clean, _ = G(noisy)
            pc = pred_clean.squeeze().cpu().numpy()
            c = clean.squeeze().cpu().numpy()
            total_psnr += psnr(pc, c)
            total_ssim += ssim(pc, c)
            # save visual comparison
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1,3, figsize=(12,4))
            ax[0].imshow(noisy.squeeze().cpu().numpy(), cmap='gray'); ax[0].set_title('noisy')
            ax[1].imshow(pc, cmap='gray'); ax[1].set_title('denoised')
            ax[2].imshow(c, cmap='gray'); ax[2].set_title('clean')
            for a in ax: a.axis('off')
            fig.savefig(os.path.join('outputs', f'compare_{i:04d}.png'))
            plt.close(fig)
            n += 1
    print('PSNR:', total_psnr/n, 'SSIM:', total_ssim/n)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    p.add_argument('--ckpt', default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    evaluate(cfg, args.ckpt)
