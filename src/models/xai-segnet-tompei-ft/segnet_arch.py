import torch
import torch.nn as nn


class SegNetEncoderBlock(nn.Module):
    """VGG-style encoder block: n_convs of Conv-BN-ReLU followed by MaxPool with index return."""
    def __init__(self, in_ch, out_ch, n_convs):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        for _ in range(n_convs - 1):
            layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                       nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        self.conv = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)


    def forward(self, x):
        x = self.conv(x)
        x, indices = self.pool(x)
        return x, indices


def _decoder_convs(in_ch, out_ch, n_convs, final=False):
    """
    Build decoder conv block: first (n_convs-1) convs keep in_ch channels,
    last conv reduces to out_ch. The very last conv of the final block has no BN/ReLU
    so it outputs raw logits.
    """
    layers = []
    for i in range(n_convs):
        c_in  = in_ch
        c_out = in_ch if i < n_convs - 1 else out_ch
        no_act = final and (i == n_convs - 1)
        layers.append(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False))
        if not no_act:
            layers += [nn.BatchNorm2d(c_out), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class SegNet(nn.Module):
    """
    SegNet (Badrinarayanan et al. 2017): symmetric VGG16 encoder-decoder.
    Decoder upsamples using stored max-pool indices — NO skip connections.
    The encoder must carry all semantic information; spatial structure is restored
    by the pooling-index unpooling only.
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder — 5 blocks matching VGG16 channel widths
        self.enc1 = SegNetEncoderBlock(in_channels, 64,  n_convs=2)
        self.enc2 = SegNetEncoderBlock(64,          128, n_convs=2)
        self.enc3 = SegNetEncoderBlock(128,         256, n_convs=3)
        self.enc4 = SegNetEncoderBlock(256,         512, n_convs=3)
        self.enc5 = SegNetEncoderBlock(512,         512, n_convs=3)

        # Shared unpooling layer (stateless — takes indices as argument)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

        # Decoder — mirrors encoder, no skip connections
        # dec5: 512 → 512 (3 convs)
        # dec4: 512 → 256 (3 convs, last conv reduces channels)
        # dec3: 256 → 128 (3 convs)
        # dec2: 128 →  64 (2 convs)
        # dec1:  64 →   K (2 convs, last conv = raw logit, no BN/ReLU)
        self.dec5 = _decoder_convs(512, 512, n_convs=3)
        self.dec4 = _decoder_convs(512, 256, n_convs=3)
        self.dec3 = _decoder_convs(256, 128, n_convs=3)
        self.dec2 = _decoder_convs(128, 64,  n_convs=2)
        self.dec1 = _decoder_convs(64, out_channels, n_convs=2, final=True)

    def forward(self, x):
        x, idx1 = self.enc1(x)
        x, idx2 = self.enc2(x)
        x, idx3 = self.enc3(x)
        x, idx4 = self.enc4(x)
        x, idx5 = self.enc5(x)

        x = self.dec5(self.unpool(x, idx5))
        x = self.dec4(self.unpool(x, idx4))
        x = self.dec3(self.unpool(x, idx3))
        x = self.dec2(self.unpool(x, idx2))
        x = self.dec1(self.unpool(x, idx1))

        return x
