import torch
import torch.nn as nn

# A compact UNet generator with dual outputs: denoised image + noise residual
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_ch, out_ch))
    def forward(self, x):
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class GeneratorUNetDual(nn.Module):
    def __init__(self, in_channels=1, base_features=32):
        super().__init__()
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features*2)
        self.down2 = Down(base_features*2, base_features*4)
        self.down3 = Down(base_features*4, base_features*8)
        self.up1 = Up(base_features*8, base_features*4)
        self.up2 = Up(base_features*4, base_features*2)
        self.up3 = Up(base_features*2, base_features)
        self.out_conv = nn.Conv2d(base_features, base_features, 3, padding=1)
        # two heads
        self.clean_head = nn.Sequential(nn.Conv2d(base_features, 1, 1), nn.Sigmoid())
        self.noise_head = nn.Conv2d(base_features, 1, 1)  # predicts residual (can be signed)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out_conv(x)
        clean = self.clean_head(x)
        noise = self.noise_head(x)
        # produce complementary outputs: clean + noise = approx input
        return clean, noise

# Patch (local) discriminator similar to PatchGAN
class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*4, 1, 4, 1, 1)  # outputs patch logits
        )
    def forward(self, x):
        return self.net(x)

# Global discriminator operates on whole image and outputs a scalar logit
class GlobalDiscriminator(nn.Module):
    def __init__(self, in_ch=1, base=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base*2, 4, 2, 1), nn.BatchNorm2d(base*2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base*2, base*4, 4, 2, 1), nn.BatchNorm2d(base*4), nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base*4, 1)
        )
    def forward(self, x):
        return self.net(x)
