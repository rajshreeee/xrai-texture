import os
import cv2
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F
import config
from pathlib import Path
SPLIT_PATH = config.SPLIT_PATH


class JointTransform:
    def __init__(self, size=(512, 512), augment=True, generator=None):
        self.size      = size
        self.augment   = augment
        self.generator = generator

    def __call__(self, image, mask):
        # Resize — PIL in, PIL out
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        mask  = F.resize(mask,  self.size, interpolation=F.InterpolationMode.NEAREST)

        if self.augment:
            # Horizontal flip
            if torch.rand(1, generator=self.generator).item() > 0.5:
                image = F.hflip(image)
                mask  = F.hflip(mask)

            # Vertical flip
            if torch.rand(1, generator=self.generator).item() > 0.5:
                image = F.vflip(image)
                mask  = F.vflip(mask)

            # ✏️ NEW: Small rotation — safe for mammography, fill=0 prevents
            # black border pixels being treated as lesion in BCE loss
            # if torch.rand(1, generator=self.generator).item() > 0.5:
            #     angle = (torch.rand(1, generator=self.generator).item() * 30) - 15  # [-15, +15]
            #     image = F.rotate(image, angle, fill=0)
            #     mask  = F.rotate(mask,  angle, fill=0)

            # ✏️ NEW: Brightness — image ONLY, not mask
            # if torch.rand(1, generator=self.generator).item() > 0.5:
            #     factor = 0.8 + torch.rand(1, generator=self.generator).item() * 0.4  # [0.8, 1.2]
            #     image  = F.adjust_brightness(image, factor)

            # # ✏️ NEW: Contrast — image ONLY, not mask
            # if torch.rand(1, generator=self.generator).item() > 0.5:
            #     factor = 0.8 + torch.rand(1, generator=self.generator).item() * 0.4  # [0.8, 1.2]
            #     image  = F.adjust_contrast(image, factor)

        return image, mask


class CancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir

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

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask  = (mask > 0).astype(np.uint8) * 255
        image = Image.fromarray(image)
        mask  = Image.fromarray(mask)

        return image, mask  # pure PIL — no ToTensor here


class TransformSubset(Dataset):
    """Applies per-split joint transform, then converts PIL → tensor once."""
    def __init__(self, dataset, indices, joint_transform=None):
        self.dataset         = dataset
        self.indices         = indices
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, mask = self.dataset[self.indices[idx]]  # PIL, PIL

        if self.joint_transform:
            image, mask = self.joint_transform(image, mask)  # PIL → PIL

        # Single ToTensor conversion — no redundant PIL↔tensor round trips
        image = transforms.ToTensor()(image)   # [3, H, W] float32 in [0, 1]
        mask  = transforms.ToTensor()(mask)    # [1, H, W] float32 in [0, 1]
        image = image_normalization(image)     # normalize image only, never mask

        return image, mask


image_normalization = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

def save_fixed_split(dataset, val_ratio=0.125, seed=42):
    """Run ONCE to generate split_indices.json. Never regenerate mid-experiment."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    split = int(len(dataset) * (1 - val_ratio))

    with open(SPLIT_PATH, "w") as f:
        json.dump({
            "train": indices[:split].tolist(),
            "val":   indices[split:].tolist()
        }, f)
    print(f"Fixed split saved: {split} train / {len(dataset)-split} val")

def load_fixed_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def make_loaders(seed, batch_size=8):
    root = '/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context'

    # Separate generators for augmentation and shuffle — both seeded per run
    # so every condition (baseline, layer2_init, etc.) sees identical data order
    # and identical augmentation decisions for the same seed
    aug_generator     = torch.Generator().manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_joint_transform = JointTransform(size=(512, 512), augment=True,  generator=aug_generator)
    val_joint_transform   = JointTransform(size=(512, 512), augment=False)
    test_joint_transform  = JointTransform(size=(512, 512), augment=False)

    # ✏️ NEW: test set lives in its own dir — completely separate from train/val
    train_images_dir = os.path.join(root, 'train/images')
    train_masks_dir  = os.path.join(root, 'train/masks')
    test_images_dir  = os.path.join(root, 'test/images')
    test_masks_dir   = os.path.join(root, 'test/masks')

    # Full dataset over train dir — split indices carve train vs val from this
    full_train_dataset = CancerDataset(train_images_dir, train_masks_dir)

    # ✏️ CHANGED: val ratio reduced from 0.20 → 0.125 (~150 val, ~1050 train)
    # split JSON must already reflect this — regenerate with create_fixed_split()
    # if needed (see config.py for SPLIT_PATH)
    if not Path(SPLIT_PATH).exists():
        save_fixed_split(full_train_dataset)
    split = load_fixed_split()   # {"train": [...], "val": [...]}

    train_set = TransformSubset(full_train_dataset, split["train"], joint_transform=train_joint_transform)
    val_set   = TransformSubset(full_train_dataset, split["val"],   joint_transform=val_joint_transform)

    # ✏️ NEW: test dataset — no split needed, uses all images in test dir
    test_dataset = CancerDataset(test_images_dir, test_masks_dir)
    test_set     = TransformSubset(test_dataset, list(range(len(test_dataset))),
                                   joint_transform=test_joint_transform)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        generator=shuffle_generator, pin_memory=True,
        drop_last=True, num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, drop_last=False, num_workers=0
    )
    # ✏️ NEW: test loader — no shuffle, no drop_last
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, drop_last=False, num_workers=0
    )

    return train_loader, val_loader, test_loader