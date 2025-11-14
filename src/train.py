"""
Train script.
Usage: python train.py --config configs/config.yaml
"""
import os
import argparse
import yaml
import glob
import numpy as np
from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from scripts.dataset import CTPairDataset
from models.networks import GeneratorUNetDual, PatchDiscriminator, GlobalDiscriminator
from utils import psnr, ssim, save_checkpoint


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def collate_fn(batch):
    noisy = torch.stack([b[0] for b in batch])
    clean = torch.stack([b[1] for b in batch])
    return noisy, clean


def train(config):
    # Device selection: Intel Arc (xpu), CUDA, or CPU
    device_str = config['training']['device']
    # Reproducibility seeds
    seed = config['training'].get('seed', 42)
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device_str == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device('xpu')
        print(f"Using Intel Arc GPU: {torch.xpu.get_device_name(0)}")
    elif device_str == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU (no GPU detected or specified)")
    
    ds = CTPairDataset(config['data']['pairs_dir'])
    n = len(ds)
    tr_n = int(n * config['data']['train_split'])
    val_n = int(n * config['data']['val_split'])
    test_n = n - tr_n - val_n
    train_set, val_set, test_set = random_split(ds, [tr_n, val_n, test_n])
    train_loader = DataLoader(train_set, batch_size=config['training']['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, collate_fn=collate_fn)

    G = GeneratorUNetDual(in_channels=1, base_features=config['model']['base_features']).to(device)
    D_local = PatchDiscriminator(in_ch=1).to(device)
    D_global = GlobalDiscriminator(in_ch=1).to(device)

    opt_G = torch.optim.Adam(G.parameters(), lr=config['training']['lr'], betas=tuple(config['training']['betas']))
    opt_Dl = torch.optim.Adam(D_local.parameters(), lr=config['training']['lr'], betas=tuple(config['training']['betas']))
    opt_Dg = torch.optim.Adam(D_global.parameters(), lr=config['training']['lr'], betas=tuple(config['training']['betas']))

    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    writer = SummaryWriter(log_dir=config['training']['log_dir'])
    best_psnr = 0.0

    for epoch in range(config['training']['epochs']):
        G.train(); D_local.train(); D_global.train()
        for i, (noisy, clean) in enumerate(train_loader):
            noisy = noisy.to(device)
            clean = clean.to(device)
            # forward
            pred_clean, pred_noise = G(noisy)
            # complementary constraint: pred_clean + pred_noise approx noisy
            recon = pred_clean + pred_noise

            # Discriminator targets
            real_label = torch.ones((clean.size(0),1,1,1), device=device)
            fake_label = torch.zeros_like(real_label)

            # Update discriminators on patches (local)
            opt_Dl.zero_grad()
            # real patches
            out_real = D_local(clean)
            out_fake = D_local(recon.detach())
            dl_loss = bce(out_real, torch.ones_like(out_real)) + bce(out_fake, torch.zeros_like(out_fake))
            dl_loss.backward(); opt_Dl.step()

            # Update global discriminator
            opt_Dg.zero_grad()
            out_real_g = D_global(clean)
            out_fake_g = D_global(recon.detach())
            dg_loss = bce(out_real_g, torch.ones_like(out_real_g)) + bce(out_fake_g, torch.zeros_like(out_fake_g))
            dg_loss.backward(); opt_Dg.step()

            # Update generator
            opt_G.zero_grad()
            # adversarial loss from both discriminators (aim to fool them)
            adv_local = bce(D_local(recon), torch.ones_like(out_fake))
            adv_global = bce(D_global(recon), torch.ones_like(out_fake_g))
            adv_loss = adv_local + adv_global
            # content loss (L1 between pred_clean and clean)
            content_loss = l1(pred_clean, clean)
            # noise-aware loss (L1 between pred_noise and (noisy - clean))
            noise_target = noisy - clean
            noise_loss = l1(pred_noise, noise_target)
            # complementary loss: encourage orthogonality between clean and noise feature maps (simple L1 on product)
            comp_loss = (pred_clean * pred_noise).abs().mean()

            g_loss = adv_loss * 0.1 + content_loss * 10.0 + noise_loss * 5.0 + comp_loss * 1.0
            g_loss.backward(); opt_G.step()

            if i % 50 == 0:
                writer.add_scalar('train/g_loss', g_loss.item(), epoch*1000 + i)
                writer.add_scalar('train/content_loss', content_loss.item(), epoch*1000 + i)
                writer.add_scalar('train/noise_loss', noise_loss.item(), epoch*1000 + i)

        # validation
        G.eval()
        val_psnr = 0.0
        val_ssim = 0.0
        nval = 0
        with torch.no_grad():
            for j, (noisy, clean) in enumerate(val_loader):
                noisy = noisy.to(device); clean = clean.to(device)
                pred_clean, _ = G(noisy)
                pc = pred_clean.squeeze().cpu().numpy()
                c = clean.squeeze().cpu().numpy()
                val_psnr += psnr(pc, c)
                val_ssim += ssim(pc, c)
                nval += 1
        val_psnr /= max(1, nval)
        val_ssim /= max(1, nval)
        writer.add_scalar('val/psnr', val_psnr, epoch)
        writer.add_scalar('val/ssim', val_ssim, epoch)
        print(f'Epoch {epoch}: val_psnr={val_psnr:.4f} val_ssim={val_ssim:.4f}')

        is_best = (val_psnr > best_psnr)
        if is_best:
            best_psnr = val_psnr
        os.makedirs(config['training']['save_dir'], exist_ok=True)
        save_checkpoint({'epoch': epoch, 'G_state': G.state_dict(), 'best_psnr': best_psnr}, is_best, filename=os.path.join(config['training']['save_dir'], f'checkpoint_epoch_{epoch}.pth.tar'))
        # Also keep a canonical best checkpoint filename for easier evaluation
        if is_best:
            torch.save({'epoch': epoch, 'G_state': G.state_dict(), 'best_psnr': best_psnr}, os.path.join(config['training']['save_dir'], 'best.pth.tar'))

    writer.close()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/config.yaml')
    args = p.parse_args()
    cfg = load_config(args.config)
    train(cfg)
