import os
import cv2
import torch
from torch import nn, optim
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import pandas as pd
# for fcbFormer
from timm.models.vision_transformer import _cfg


# visualization

import matplotlib.pyplot as plt
from PIL import Image
import imageio

import optuna
import neptune.integrations.optuna as npt_utils
from neptune.integrations.optuna import NeptuneCallback
from monai.losses import DiceLoss

import logging
import json

import sys

#ViT
from torchvision.models import vit_b_16
import torch.nn.functional as F

#Swin
from torchvision.models.swin_transformer import swin_t

from torchsummary import summary

import os
import torch
import numpy as np
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage

import os
import torch
import timm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time
import matplotlib.pyplot as plt
from torch.utils.data import ConcatDataset
from albumentations.pytorch import ToTensorV2
from skmultilearn.model_selection import IterativeStratification
from torchvision import models
from collections import Counter
import torch.nn.functional as F
from sklearn.utils.class_weight import compute_class_weight


batch_size = 16

class RB(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.3):
        super().__init__()

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.Dropout(dropout_prob)  # Dropout
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),  # Batch Normalization
            nn.Dropout(dropout_prob)  # Dropout
        )

        if out_channels == in_channels:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = self.in_layers(x)
        h = self.out_layers(h)
        return h + self.skip(x)


class FCB(nn.Module):
    def __init__(
        self,
        in_channels=3,
        min_level_channels=32,
        min_channel_mults=[1, 1, 2, 2, 3],
        n_levels_down=5,
        n_levels_up=5,
        n_RBs=1,
        in_resolution=512,
        dropout_prob=0.3
    ):
        super().__init__()

        self.enc_blocks = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(in_channels, min_level_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(min_level_channels),  # Batch Normalization
                nn.Dropout(dropout_prob)  # Dropout
            )]
        )
        ch = min_level_channels
        enc_block_chans = [min_level_channels]
        for level in range(n_levels_down):
            min_channel_mult = min_channel_mults[level]
            for block in range(n_RBs):
                self.enc_blocks.append(
                    nn.Sequential(
                        RB(ch, min_channel_mult * min_level_channels, dropout_prob),
                        nn.BatchNorm2d(min_channel_mult * min_level_channels),  # Batch Normalization
                        nn.Dropout(dropout_prob)  # Dropout
                    )
                )
                ch = min_channel_mult * min_level_channels
                enc_block_chans.append(ch)
            if level != n_levels_down - 1:
                self.enc_blocks.append(
                    nn.Sequential(
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1, stride=2),
                        nn.BatchNorm2d(ch),  # Batch Normalization
                        nn.Dropout(dropout_prob)  # Dropout
                    )
                )
                enc_block_chans.append(ch)

        self.middle_block = nn.Sequential(
            RB(ch, ch, dropout_prob),
            nn.BatchNorm2d(ch),  # Batch Normalization
            nn.Dropout(dropout_prob),  # Dropout
            RB(ch, ch, dropout_prob),
            nn.BatchNorm2d(ch),  # Batch Normalization
            nn.Dropout(dropout_prob)  # Dropout
        )

        self.dec_blocks = nn.ModuleList([])
        for level in range(n_levels_up):
            min_channel_mult = min_channel_mults[::-1][level]

            for block in range(n_RBs + 1):
                layers = [
                    RB(
                        ch + enc_block_chans.pop(),
                        min_channel_mult * min_level_channels,
                        dropout_prob
                    ),
                    nn.BatchNorm2d(min_channel_mult * min_level_channels),  # Batch Normalization
                    nn.Dropout(dropout_prob)  # Dropout
                ]
                ch = min_channel_mult * min_level_channels
                if level < n_levels_up - 1 and block == n_RBs:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                            nn.BatchNorm2d(ch),  # Batch Normalization
                            nn.Dropout(dropout_prob)  # Dropout
                        )
                    )
                self.dec_blocks.append(nn.Sequential(*layers))

    def forward(self, x):
        hs = []
        h = x
        for module in self.enc_blocks:
            h = module(h)
            hs.append(h)
        h = self.middle_block(h)
        for module in self.dec_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in)
        return h


class TB(nn.Module):
    def __init__(self, dropout_prob=0.3):
        super().__init__()
        # Load Vision Transformer (ViT-B/16) as the encoder
        self.backbone = vit_b_16(weights="DEFAULT")
        
        # Remove the classification head (VIT)
        self.backbone.heads.head = nn.Identity()
        
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Progressive Locality Decoder (PLD+)
        self.LE = nn.ModuleList([
            nn.Sequential(
                RB(768, 64, dropout_prob),  # Residual Block (RB)
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob),  # Dropout
                RB(64, 64, dropout_prob),
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob),  # Dropout
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
            ) for _ in range(4)
        ])

        self.SFA = nn.ModuleList([
            nn.Sequential(
                RB(128, 64, dropout_prob),  # Residual Block (RB)
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob),  # Dropout
                RB(64, 64, dropout_prob),
                nn.BatchNorm2d(64),  # Batch Normalization
                nn.Dropout(dropout_prob)  # Dropout
            ) for _ in range(3)
        ])

    def get_pyramid(self, x):
        """
        Extract feature pyramid from Vision Transformer (ViT) encoder.
        """
        # Forward pass through ViT
        x = self.backbone(x)  # Output shape: [B, 768]

        # Reshape to [B, 768, 1, 1] (since ViT outputs a 1D feature vector)
        B, C = x.shape
        x = x.reshape(B, C, 1, 1)

        # Upsample to [B, 768, 14, 14] to match the expected spatial dimensions
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=False)

        # Repeat the same feature map for all levels (since ViT is non-hierarchical)
        pyramid = [x] * 4
        return pyramid

    def forward(self, x):
        """
        Forward pass for the Transformer Branch (TB) with Vision Transformer (ViT).
        """
        # Ensure the input is 4D: [batch_size, channels, height, width]
        if x.dim() == 2:  # If input is 2D [height, width]
            x = x.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        elif x.dim() == 3:  # If input is 3D [channels, height, width]
            x = x.unsqueeze(0)  # Add batch dimension

        # Extract feature pyramid from ViT encoder
        pyramid = self.get_pyramid(x)

        # Apply Local Emphasis (LE) modules to each level of the pyramid
        pyramid_emph = []
        for i, level in enumerate(pyramid):
            pyramid_emph.append(self.LE[i](pyramid[i]))

        # Stepwise Feature Aggregation (SFA)
        l_i = pyramid_emph[-1]
        for i in range(2, -1, -1):
            l = torch.cat((pyramid_emph[i], l_i), dim=1)
            l = self.SFA[i](l)
            l_i = l

        return l_i


class FCBFormer(nn.Module):
    def __init__(self, size=512, dropout_prob=0.3):
        super().__init__()
        self.TB = TB(dropout_prob=dropout_prob)  # Transformer Branch (ViT)
        self.FCB = FCB(in_resolution=size, dropout_prob=dropout_prob)  # Fully Convolutional Branch
        self.PH = nn.Sequential(
            RB(64 + 32, 64, dropout_prob),  # Residual Block
            RB(64, 64, dropout_prob),
            nn.Conv2d(64, 1, kernel_size=1)  # Final prediction layer
        )
        self.up_tosize = nn.Upsample(size=size)  # Upsample to full resolution

    def forward(self, x):
        try:
            x1 = self.TB(x)  # Transformer Branch output
            x2 = self.FCB(x)  # Fully Convolutional Branch output
            x1 = self.up_tosize(x1)
            if x2.shape[2:] != (512, 512):
                x2 = F.interpolate(x2, size=(512, 512), mode='bilinear', align_corners=False)
            x = torch.cat((x1, x2), dim=1)
            out = self.PH(x)
            return out
        except Exception as e:
            raise Exception('An exception occurred during forward pass') from e        


def dice_score(preds, targets, smooth=1e-6):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    return (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)


# -------- Dataset --------
class CancerDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 masks_dir: str,
                 csv_file:   str,
                 image_transform=None,
                 mask_transform=None):
        self.images_dir     = images_dir
        self.masks_dir      = masks_dir
        self.image_transform = image_transform
        self.mask_transform  = mask_transform

        # Load and filter annotation CSV
        df = pd.read_csv(csv_file)
        df['pathology'] = df['pathology'].str.lower().replace('benign_without_callback', 'benign')
        valid_path = {"malignant", "benign"}
        valid_shape = {"round", "oval", "lobulated", "irregular"}
        valid_birads = {2, 3, 4, 5}

        ann = df[
            (df['pathology'].isin(valid_path)) &
            (df['mass shape'].str.lower().isin(valid_shape)) &
            (df['assessment'].isin(valid_birads))
        ].reset_index(drop=True)

        self.path_map   = {"benign": 0, "malignant": 1}
        self.shape_map  = {"round": 0, "oval": 1, "lobulated": 2, "irregular": 3}
        self.birads_map = {2: 0, 3: 1, 4: 2, 5: 3}

        # Store annotations as lookup by stem
        self.ann_lookup = {
            row['image_name']: row
            for _, row in ann.iterrows()
        }

        self.valid_stems = set(self.ann_lookup.keys())

        # Keep only files whose stem starts with an annotated stem
        exts = ('.jpg', '.jpeg', '.png')
        self.image_files = []
        for f in os.listdir(images_dir):
            if not f.lower().endswith(exts):
                continue
            file_stem = os.path.splitext(f)[0]
            for valid_stem in self.valid_stems:
                if file_stem.startswith(valid_stem):
                    self.image_files.append(f)
                    break

        self.image_files = sorted(self.image_files)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        full_stem = os.path.splitext(img_name)[0]

        # Find matching annotation row using prefix match
        matching_stem = None
        for stem in self.valid_stems:
            if full_stem.startswith(stem):
                matching_stem = stem
                break

        if matching_stem is None:
            raise ValueError(f"No matching annotation for {full_stem}")

        row = self.ann_lookup[matching_stem]

        # Load image and mask
        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, img_name)  # same filename

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        image = Image.open(img_path).convert('RGB')
        mask  = Image.open(mask_path).convert('L')

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            mask = (mask > 0).float()

        # Get labels
        p = self.path_map[row['pathology']]
        s = self.shape_map[row['mass shape'].lower()]
        b = self.birads_map[int(row['assessment'])]
        labels = torch.tensor([p, s, b], dtype=torch.long)

        return image, mask, labels

# -------- Transforms --------
image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])
mask_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),  # gives [1,H,W] in [0,1]
])

# -------- Merge Datasets --------
train_imgs = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_images"
train_masks= "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/train/aug_masks"
test_imgs  = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/aug_images"
test_masks = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/test/aug_masks"
csv_file = "/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context/CBIS_DDSM_PATCHED_ANNOTATIONS_CONTEXT.csv"

# train_dataset = CancerDataset(train_imgs, train_masks, csv_file, image_transform, mask_transform)
# test_dataset_extra  = CancerDataset(test_imgs,  test_masks, csv_file,  image_transform, mask_transform)
# combined_dataset = ConcatDataset([train_dataset, test_dataset_extra])

# # 1. Prepare multilabel matrix
# all_labels = []
# for i in range(len(combined_dataset)):
#     _, labels = combined_dataset[i]
#     all_labels.append(labels)

# # Stack into (n_samples, 3) [pathology, shape, birads]
# all_labels_np = torch.stack(all_labels).numpy()

# print("[INFO] Overall dataset size:", len(all_labels_np))

# pathology_counts = Counter(all_labels_np[:, 0])
# shape_counts = Counter(all_labels_np[:, 1])
# birads_counts = Counter(all_labels_np[:, 2])

# print("Pathology distribution (0=benign,1=malignant):", dict(pathology_counts))
# print("Shape distribution (0=round,1=oval,2=lobulated,3=irregular):", dict(shape_counts))
# print("BIRADS distribution (0->B2,1->B3,2->B4,3->B5):", dict(birads_counts))

# # 2. Setup stratifier
# stratifier = IterativeStratification(n_splits=5, order=1)  # 5 means 80-20 split (1 fold test)

# # 3. Get train/test split
# train_idx, temp_idx = next(stratifier.split(np.zeros(len(all_labels_np)), all_labels_np))

# # 4. Further split temp into val and test (50-50)
# temp_labels_np = all_labels_np[temp_idx]
# stratifier_val_test = IterativeStratification(n_splits=2, order=1)
# val_idx, test_idx = next(stratifier_val_test.split(np.zeros(len(temp_labels_np)), temp_labels_np))

# # Need to adjust temp_idx mapping
# val_idx = temp_idx[val_idx]
# test_idx = temp_idx[test_idx]

# # 5. Print distributions after split
# def print_split_distribution(indices, name):
#     labels = all_labels_np[indices]
#     pathology_counts = Counter(labels[:, 0])
#     shape_counts = Counter(labels[:, 1])
#     birads_counts = Counter(labels[:, 2])
#     print(f"\n[INFO] {name} size: {len(indices)}")
#     print("Pathology distribution:", dict(pathology_counts))
#     print("Shape distribution:", dict(shape_counts))
#     print("BIRADS distribution:", dict(birads_counts))

imgs = os.listdir(train_imgs)
img_stems = {os.path.splitext(f)[0] for f in imgs}

csv = pd.read_csv(csv_file)
csv_stems = set(csv['image_name'].astype(str))

print(f"Image stems found in folder: {len(img_stems)}")
print(f"Image names in CSV: {len(csv_stems)}")
print(f"Intersection: {len(img_stems & csv_stems)}")


ds1 = CancerDataset(train_imgs, train_masks, csv_file, image_transform, mask_transform)
ds2 = CancerDataset(test_imgs,  test_masks,  csv_file, image_transform, mask_transform)
combined = ConcatDataset([ds1, ds2])


# collect label vectors for every sample
all_labels = np.stack([
    combined[i][2].numpy()
    for i in range(len(combined))
])

# iterative stratification exactly as in your classification flow:
strat = IterativeStratification(n_splits=5, order=1)
train_idx, temp_idx = next(strat.split(np.zeros(len(all_labels)), all_labels))

strat_vt = IterativeStratification(n_splits=2, order=1)
temp_labels = all_labels[temp_idx]
val_idx, test_idx = next(strat_vt.split(np.zeros(len(temp_labels)), temp_labels))
val_idx  = temp_idx[val_idx]
test_idx = temp_idx[test_idx]

def print_split_distribution(indices, name):
    labels = all_labels[indices]
    pathology_counts = Counter(labels[:, 0])
    shape_counts = Counter(labels[:, 1])
    birads_counts = Counter(labels[:, 2])
    print(f"\n[INFO] {name} size: {len(indices)}")
    print("Pathology distribution:", dict(pathology_counts))
    print("Shape distribution:", dict(shape_counts))
    print("BIRADS distribution:", dict(birads_counts))


print_split_distribution(train_idx, "TRAIN")
print_split_distribution(val_idx,   "VALID")
print_split_distribution(test_idx,  "TEST")

# 6. Final DataLoaders
train_loader = DataLoader(Subset(combined, train_idx), batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
val_loader = DataLoader(Subset(combined, val_idx), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
test_loader = DataLoader(Subset(combined, test_idx), batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

print(f"\nFinal Split Sizes --> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

# -------- DataLoaders --------

# -------- Model, Losses, Optimizer --------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FCBFormer(size=512).to(device)

# Loss = BCE + Dice
bce_loss = nn.BCEWithLogitsLoss()
def dice_loss(preds, targets, smooth=1e-6):
    preds = torch.sigmoid(preds)
    preds = preds.view(-1)
    targets = targets.view(-1)
    inter = (preds * targets).sum()
    return 1 - ((2*inter + smooth)/(preds.sum() + targets.sum() + smooth))

def seg_loss(logits, masks):
    return bce_loss(logits, masks) + dice_loss(logits, masks)

optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# -------- Training Loop --------
num_epochs = 30
best_val = float('inf')
early_stop_cnt = 0
patience=5

train_losses, val_losses = [], []

for ep in range(1, num_epochs+1):
    # -- train --
    model.train()
    running = 0
    for imgs, masks, _ in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        logits = model(imgs)
        loss = seg_loss(logits, masks)
        loss.backward()
        optimizer.step()
        running += loss.item()
    train_losses.append(running/len(train_loader))

    # -- val --
    model.eval()
    running = 0
    with torch.no_grad():
        for imgs, masks, _ in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            running += seg_loss(logits, masks).item()
    val_losses.append(running/len(val_loader))
    torch.cuda.empty_cache()
    scheduler.step(val_losses[-1])

    print(f"Epoch {ep}/{num_epochs}  Train Loss: {train_losses[-1]:.4f}  Val Loss: {val_losses[-1]:.4f}")

    # early stopping
    if val_losses[-1] < best_val:
        best_val = val_losses[-1]
        early_stop_cnt = 0
        torch.save(model.state_dict(), "saads_segmentation.pth")
    else:
        early_stop_cnt += 1
        if early_stop_cnt >= patience:
            print(f"Early stopping at epoch {ep}")
            break

# -------- Test Evaluation --------
model.load_state_dict(torch.load("saads_segmentation.pth"))
model.eval()
dice_scores = []
with torch.no_grad():
    for imgs, masks, _ in test_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        # compute per‐batch dice
        bs = preds.shape[0]
        for i in range(bs):
            dice_scores.append(dice_loss(preds[i:i+1], masks[i:i+1]).item())
print("Average Test Dice:", 1 - np.mean(dice_scores))

# -------- Plot Loss Curves --------
plt.figure(figsize=(10,4))
plt.plot(train_losses, label="Train")
plt.plot(val_losses,   label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("saads_fcb_loss_curves.png")



