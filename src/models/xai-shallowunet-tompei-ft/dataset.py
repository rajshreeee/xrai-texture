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
        image = F.resize(image, self.size, interpolation=F.InterpolationMode.BILINEAR)
        mask  = F.resize(mask,  self.size, interpolation=F.InterpolationMode.NEAREST)

        if self.augment:
            max_shift = int(0.10 * self.size[0])
            tx = torch.randint(-max_shift, max_shift + 1, (1,),
                               generator=self.generator).item()
            ty = torch.randint(-max_shift, max_shift + 1, (1,),
                               generator=self.generator).item()
            image = F.affine(image, angle=0, translate=[tx, ty], scale=1.0, shear=0,
                             interpolation=F.InterpolationMode.BILINEAR, fill=0)
            mask  = F.affine(mask,  angle=0, translate=[tx, ty], scale=1.0, shear=0,
                             interpolation=F.InterpolationMode.NEAREST,  fill=0)
            if torch.rand(1, generator=self.generator).item() > 0.5:
                image = F.hflip(image)
                mask  = F.hflip(mask)
            if torch.rand(1, generator=self.generator).item() > 0.5:
                image = F.vflip(image)
                mask  = F.vflip(mask)

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

        image = transforms.ToTensor()(image)   # [3, H, W] float32 in [0, 1]
        mask  = transforms.ToTensor()(mask)    # [1, H, W] float32 in [0, 1]
        image = image_normalization(image)

        return image, mask


# Grayscale-aware: mammograms have R=G=B, so identical per-channel values
# prevent artificial colour gradients that would corrupt texture kernel responses.
# Values are the channel-wise average of standard ImageNet stats.
image_normalization = transforms.Normalize(
    mean=[0.449, 0.449, 0.449],
    std=[0.226, 0.226, 0.226]
)

def save_fixed_split(dataset, val_ratio=0.2, seed=42):
    """Split at original-image level so _aug* copies never cross the train/val boundary."""
    import re
    rng = np.random.default_rng(seed)

    originals = np.array([name for name in dataset.images if not re.search(r'_aug\d+', name)])
    rng.shuffle(originals)
    split_idx = int(len(originals) * (1 - val_ratio))
    train_orig = set(originals[:split_idx])
    val_orig   = set(originals[split_idx:])

    train_indices, val_indices = [], []
    for idx, name in enumerate(dataset.images):
        base = re.sub(r'_aug\d+', '', os.path.splitext(name)[0]) + os.path.splitext(name)[1]
        if name in val_orig or base in val_orig:
            val_indices.append(idx)
        else:
            train_indices.append(idx)

    with open(SPLIT_PATH, "w") as f:
        json.dump({"train": train_indices, "val": val_indices}, f)
    print(f"Fixed split saved: {len(train_indices)} train / {len(val_indices)} val")

def load_fixed_split():
    with open(SPLIT_PATH) as f:
        return json.load(f)


def make_loaders(seed, batch_size=8):
    root = config.DATA_ROOT

    aug_generator     = torch.Generator().manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_joint_transform = JointTransform(size=(512, 512), augment=True,  generator=aug_generator)
    val_joint_transform   = JointTransform(size=(512, 512), augment=False)
    test_joint_transform  = JointTransform(size=(512, 512), augment=False)

    train_images_dir = os.path.join(root, 'train/images')
    train_masks_dir  = os.path.join(root, 'train/masks')
    test_images_dir  = os.path.join(root, 'test/images')
    test_masks_dir   = os.path.join(root, 'test/masks')

    full_train_dataset = CancerDataset(train_images_dir, train_masks_dir)

    if not Path(SPLIT_PATH).exists():
        save_fixed_split(full_train_dataset)
    split = load_fixed_split()

    train_set = TransformSubset(full_train_dataset, split["train"], joint_transform=train_joint_transform)
    val_set   = TransformSubset(full_train_dataset, split["val"],   joint_transform=val_joint_transform)

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
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        pin_memory=True, drop_last=False, num_workers=0
    )

    return train_loader, val_loader, test_loader
