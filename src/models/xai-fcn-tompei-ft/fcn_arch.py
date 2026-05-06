import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN8sEncoderBlock(nn.Module):
    """VGG-style encoder block: n_convs of Conv-BN-ReLU followed by MaxPool (no index return)."""
    def __init__(self, in_ch, out_ch, n_convs):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        for _ in range(n_convs - 1):
            layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x)


class FCN8s(nn.Module):
    """
    FCN-8s with VGG-style encoder identical to SegNet (enc1-enc5).
    Skip connections from pool3 (256-ch), pool4 (512-ch), pool5 (512-ch).

    For 512×512 input:
      pool3 → 64×64, pool4 → 32×32, pool5 → 16×16
    All upsample dimensions align exactly without cropping.

    enc1.conv[0] attribute path is identical to SegNet, so inject_kernels() works unchanged.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        self.enc1 = FCN8sEncoderBlock(in_channels, 64,  n_convs=2)
        self.enc2 = FCN8sEncoderBlock(64,          128, n_convs=2)
        self.enc3 = FCN8sEncoderBlock(128,         256, n_convs=3)
        self.enc4 = FCN8sEncoderBlock(256,         512, n_convs=3)
        self.enc5 = FCN8sEncoderBlock(512,         512, n_convs=3)

        self.score_pool5 = nn.Conv2d(512, out_channels, 1)
        self.score_pool4 = nn.Conv2d(512, out_channels, 1)
        self.score_pool3 = nn.Conv2d(256, out_channels, 1)

        # Separate learned upsampling at each scale — independent filters, better expressiveness.
        self.upsample_5to4 = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1, bias=False)
        self.upsample_4to3 = nn.ConvTranspose2d(out_channels, out_channels, 4, stride=2, padding=1, bias=False)
        self.upsample_8x   = nn.ConvTranspose2d(out_channels, out_channels, 16, stride=8, padding=4, bias=False)

    def forward(self, x):
        p1 = self.enc1(x)
        p2 = self.enc2(p1)
        p3 = self.enc3(p2)
        p4 = self.enc4(p3)
        p5 = self.enc5(p4)

        s5 = self.score_pool5(p5)
        s4 = self.score_pool4(p4) + self.upsample_5to4(s5)
        s3 = self.score_pool3(p3) + self.upsample_4to3(s4)
        return self.upsample_8x(s3)
