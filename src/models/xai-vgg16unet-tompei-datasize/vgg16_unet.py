import torch
import torch.nn as nn


class VGGBlock(nn.Module):
    """Conv block with 2 or 3 Conv-BN-ReLU units.

    self.conv[0].weight is always the first Conv2d — the injection target.
    This mirrors DoubleConv from vanilla_unet.py so inject_kernels() is unchanged.
    """
    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            ic = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv2d(ic, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*layers)
        nn.init.dirac_(self.conv[0].weight)

    def forward(self, x):
        return self.conv(x)


class VGG16UNet(nn.Module):
    """VGG16-style encoder + U-Net decoder for binary segmentation.

    Encoder follows VGG16 channel progression (64, 128, 256, 512, 512) with
    2-conv blocks for stages 1-2 and 3-conv blocks for stages 3-5, matching
    the original VGG16 architecture.

    Kernel injection site: enc1.conv[0].weight  (shape: 64 × 3 × 3 × 3)
    — identical access pattern to VanillaUNet's enc1.conv[0].weight.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder — VGG16-style
        self.enc1 = VGGBlock(in_channels, 64,  n_convs=2)   # injection site
        self.enc2 = VGGBlock(64,          128, n_convs=2)
        self.enc3 = VGGBlock(128,         256, n_convs=3)
        self.enc4 = VGGBlock(256,         512, n_convs=3)
        self.enc5 = VGGBlock(512,         512, n_convs=3)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = VGGBlock(512, 1024, n_convs=2)

        # Decoder — 5 stages (one per encoder block)
        self.up5  = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec5 = VGGBlock(1024, 512)   # 512 (up) + 512 (s5)

        self.up4  = nn.ConvTranspose2d(512, 512, 2, stride=2)
        self.dec4 = VGGBlock(1024, 256)   # 512 (up) + 512 (s4)

        self.up3  = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = VGGBlock(512, 128)    # 256 (up) + 256 (s3)

        self.up2  = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = VGGBlock(256, 64)     # 128 (up) + 128 (s2)

        self.up1  = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = VGGBlock(128, 64)     # 64  (up) + 64  (s1)

        self.out_conv = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder — skip connections taken after each block, before pool
        s1 = self.enc1(x)                     # H×W,   64ch
        s2 = self.enc2(self.pool(s1))          # H/2,   128ch
        s3 = self.enc3(self.pool(s2))          # H/4,   256ch
        s4 = self.enc4(self.pool(s3))          # H/8,   512ch
        s5 = self.enc5(self.pool(s4))          # H/16,  512ch

        b = self.bottleneck(self.pool(s5))     # H/32, 1024ch

        # Decoder — upsample → concat skip → conv block
        x = self.dec5(torch.cat([self.up5(b), s5], dim=1))   # H/16, 512ch
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))   # H/8,  256ch
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))   # H/4,  128ch
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))   # H/2,   64ch
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))   # H,     64ch

        return self.out_conv(x)
