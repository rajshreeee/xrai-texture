import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16
import os
import pandas as pd

#Working sort of
# class FixedConv2d(nn.Module):
    
#     def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
#         super().__init__()
#         base_filter = torch.tensor([
#             [-1, -4, -6, -4, -1],
#             [-2, -8, -12, -8, -2],
#             [0, 0, 0, 0, 0],
#             [2, 8, 12, 8, 2],
#             [1, 4, 6, 4, 1]
#         ], dtype=torch.float32)

#         self.weight = nn.Parameter(
#             base_filter.expand(out_channels, in_channels, 5, 5) / (in_channels * out_channels),
#             requires_grad=False
#         )

#         self.stride = stride
#         self.padding = padding
#         print(f"FixedConv2d initialized with kernel shape: {self.weight.shape}")
#         print(self.weight[0, 0])  

#     def forward(self, x):
#         return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)


class FixedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        base_filter = torch.tensor([
            [-1, -4, -6, -4, -1],
            [-2, -8, -12, -8, -2],
            [0, 0, 0, 0, 0],
            [2, 8, 12, 8, 2],
            [1, 4, 6, 4, 1]
        ], dtype=torch.float32)

        # Safe repeat
        kernel = base_filter.view(1, 1, 5, 5).repeat(out_channels, in_channels, 1, 1)
        print(f"FixedConv2d initialized with kernel shape: {kernel.shape}")
        print(kernel[0, 0])
        # Optional: normalize
        # denom = max(in_channels * out_channels, 1)
        # kernel = kernel / denom

        self.weight = nn.Parameter(kernel, requires_grad=True)
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.stride, padding=self.padding)



def save_kernels_compact_csv(model, epoch, save_dir='kernel_logs'):
    os.makedirs(save_dir, exist_ok=True)
    rows = []

    for name, module in model.named_modules():
        if isinstance(module, FixedConv2d):
            # Assume all kernels are identical, just grab the first one
            kernel = module.weight[0, 0].detach().cpu()

            kernel_str = '\n'.join([','.join([f'{v}' for v in row]) for row in kernel])

            rows.append({
                'layer': name,
                'epoch': epoch,
                'kernel_matrix': kernel_str
            })

    # Save to CSV
    df = pd.DataFrame(rows)
    path = os.path.join(save_dir, f'epoch_{epoch}_kernels.csv')
    df.to_csv(path, index=False)
    print(f"[Epoch {epoch}] Clean kernel snapshot saved to {path}")


class FixedSegNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.vit = vit_b_16(weights="DEFAULT")
        self.vit.heads = nn.Identity()

        for param in self.vit.parameters():
            param.requires_grad = False  # 🔒 freeze encoder

        self.hidden_dim = self.vit.hidden_dim  # 768

        # Decoder with fixed Law filters
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = FixedConv2d(in_channels=self.hidden_dim, out_channels=256)
        # self.conv1 = nn.Conv2d(in_channels=self.hidden_dim, out_channels=256,kernel_size=5,padding=2)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = FixedConv2d(in_channels=256,  out_channels=128)
        # self.conv2 = nn.Conv2d(in_channels=256,  out_channels=128,kernel_size=5,padding=2)

        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = FixedConv2d(in_channels=128,  out_channels=64)
        # self.conv3 = nn.Conv2d(in_channels=128,  out_channels=64,kernel_size=5,padding=2)

        self.final_conv = nn.Conv2d(in_channels=64,  out_channels=num_classes, kernel_size=1)
        self.upsample_output = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)

    def forward(self, x):
        B, _, _, _ = x.shape

        # Resize input to ViT size
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Get feature embedding: [B, 768]
        feats = self.vit(x)  # [B, 768]
        x = feats.view(B, self.hidden_dim, 1, 1)  # [B, 768, 1, 1]

        x = self.up1(x)  # 2×2
        x = self.conv1(x)

        x = self.up2(x)  # 4×4
        x = self.conv2(x)

        x = self.up3(x)  # 8×8
        x = self.conv3(x)

        x = self.final_conv(x)  # [B, 1, 8, 8]
        x = self.upsample_output(x)  # [B, 1, 512, 512]
        return x



