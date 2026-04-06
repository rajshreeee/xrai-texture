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

class ResizeTransform:
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        return F.resize(img, self.size)


# --- Reuse your existing CancerDataset unchanged ---
class CancerDataset(Dataset):
    def __init__(self, images_dir, masks_dir, image_transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        valid_img_extensions = (".jpg", ".jpeg", ".png")
        self.images = sorted([f for f in os.listdir(images_dir)
                               if f.endswith(valid_img_extensions)])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_name = self.images[idx]
        image_path = os.path.join(self.images_dir, image_name)
        mask_path  = os.path.join(self.masks_dir, image_name.split('.')[0] + '.png')

        image = cv2.imread(image_path)
        mask  = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = (mask > 0).astype(np.uint8) * 255

        image = Image.fromarray(image)
        mask  = Image.fromarray(mask)

        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


# --- Transforms (keep consistent with your existing pipeline) ---
image_transform = transforms.Compose([
    ResizeTransform((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    ResizeTransform((512, 512)),
    transforms.ToTensor()
])


# --- Fixed split: save once, reuse across all runs ---
SPLIT_PATH = "/ediss_data/ediss2/xai-texture/src/models/xai-model/data/split_indices.json"  # fixed split saved once

def save_fixed_split(dataset, val_ratio=0.2, seed=42):
    """Call this ONCE before any training run."""
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


# --- DataLoader factory used by each training run ---
def make_loaders(seed, batch_size=4):
    CBIS_DDSM_Patches_Mass_Context = '/ediss_data/ediss2/xai-texture/data/CBIS_DDSM_Patches_Mass_Context'
    images_dir = CBIS_DDSM_Patches_Mass_Context + '/train/images'
    masks_dir  = CBIS_DDSM_Patches_Mass_Context + '/train/masks'

    dataset = CancerDataset(images_dir, masks_dir,
                             image_transform=image_transform,
                             mask_transform=mask_transform)

    # Save split on first call only
    if not Path(SPLIT_PATH).exists():
        save_fixed_split(dataset)

    split = load_fixed_split()
    train_set = Subset(dataset, split["train"])
    val_set   = Subset(dataset, split["val"])

    # Seed the shuffle generator per run
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, generator=g, pin_memory=True,
                              drop_last=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size,
                              shuffle=False, pin_memory=True, drop_last=True)

    return train_loader, val_loader