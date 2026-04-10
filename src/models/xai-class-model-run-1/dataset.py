import os
import cv2
import json
import numpy as np
from PIL import Image
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F


import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
from torchvision import transforms

class JointTransform:
    def __init__(self, size=(512, 512), augment=True, generator=None):
        self.size = size
        self.augment = augment
        self.generator = generator

    def __call__(self, image, mask):
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        mask  = F.resize(mask,  self.size, interpolation=F.InterpolationMode.NEAREST)

        if self.augment:
            if torch.rand(1, generator=self.generator).item() > 0.5:
                image = F.hflip(image)
                mask  = F.hflip(mask)

            if torch.rand(1, generator=self.generator).item() > 0.5:
                image = F.vflip(image)
                mask  = F.vflip(mask)

        return image, mask

# ----------------------------
# Dataset
# ----------------------------
class CancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir,
                 joint_transform=None,
                 image_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.joint_transform = joint_transform
        self.image_transform = image_transform

        valid_ext = (".jpg", ".jpeg", ".png")
        self.images = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(valid_ext)
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]

        image_path = os.path.join(self.images_dir, image_name)
        mask_path  = os.path.join(self.masks_dir,
                                  os.path.splitext(image_name)[0] + ".png")

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        # Convert mask to binary (0,1)
        mask = (mask > 0).astype(np.uint8) * 255

        # Convert to PIL
        image = Image.fromarray(image)
        mask  = Image.fromarray(mask)

        # Apply joint transform (sync augmentations)
        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)

        # Convert to tensor
        image = transforms.ToTensor()(image)
        mask  = transforms.ToTensor()(mask)

        # Normalize image ONLY
        if self.image_transform:
            image = self.image_transform(image)

        return image, mask


# ----------------------------
# Transforms
# ----------------------------

image_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)


# ----------------------------
# DataLoader factory
# ----------------------------
def make_loaders(seed, batch_size=8):
    root = '/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context'
    
    # One generator controls BOTH shuffle order AND augmentation decisions
    aug_generator = torch.Generator()
    aug_generator.manual_seed(seed)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)

    train_images_dir = os.path.join(root, 'train/images')
    train_masks_dir  = os.path.join(root, 'train/masks')

    val_images_dir = os.path.join(root, 'test/images')
    val_masks_dir  = os.path.join(root, 'test/masks')

    train_joint_transform = JointTransform(
        size=(512, 512), augment=True, generator=aug_generator
    )
    val_joint_transform = JointTransform(
        size=(512, 512), augment=False  # no generator needed, no randomness
    )

    train_set = CancerDataset(
        train_images_dir,train_masks_dir,
        joint_transform=train_joint_transform,
        image_transform=image_normalization
    )

    val_set = CancerDataset(
        val_images_dir, val_masks_dir,
        joint_transform=val_joint_transform,
        image_transform=image_normalization
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
        pin_memory=True,
        drop_last=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )

    return train_loader, val_loader
