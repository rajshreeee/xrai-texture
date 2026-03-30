import torch
import torch.nn as nn
import numpy as np
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.ndimage import zoom
import wandb
import json
from pathlib import Path
from dataset import make_loaders
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import cv2
from logger import setup_logger

SEEDS = [42, 123, 256, 789, 1024]
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPLIT_INDICES_PATH = "/ediss_data/ediss2/xai-texture/src/models/xai-model/data/split_indices.json"  # fixed split saved once
CHECKPOINT_DIR = Path("/ediss_data/ediss2/xai-texture/src/models/xai-model/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

kernels = {
    "layer3": [
        np.array([
            [-0.66570723, -0.12788272, -0.09561324],
            [-0.06334380,  0.48523712,  0.45296767],
            [-0.23544757,  0.13027307,  0.11951658]
        ], dtype=np.float32),

        np.array([
            [-0.45346430,  0.10560124, -0.19256702],
            [-0.09007170,  0.49694710,  0.38513404],
            [-0.54664180,  0.13355458,  0.16150787]
        ], dtype=np.float32)
    ],

    "layer4": [
        np.array([
            [-0.70573944, -0.18723700, -0.10081992],
            [-0.18723700,  0.29885903,  0.33126545],
            [-0.10081992,  0.33126545,  0.32046336]
        ], dtype=np.float32),

        np.array([
            [-0.70573944, -0.18723700, -0.10081992],
            [-0.18723700,  0.29885903,  0.33126545],
            [-0.10081992,  0.33126545,  0.32046336]
        ], dtype=np.float32)
    ],

    "classifier": [
        np.array([
            [-0.42984945, -0.04796607, -0.42154756],
            [-0.04796607,  0.52485900,  0.27580470],
            [-0.37173668,  0.32561558,  0.19278659]
        ], dtype=np.float32),

        np.array([
            [-0.74325246, -0.07556172, -0.17447884],
            [-0.07556172,  0.44375327,  0.29537760],
            [-0.11265561,  0.29537760,  0.14700192]
        ], dtype=np.float32)
    ]
}

def build_model(seed, inject_layer=None):
    """
    inject_layer: None (baseline), 'layer2', or 'layer3'
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )

    if inject_layer is not None:
        kernels_3x3 = kernels  # 9 kernels
        inject_kernels(model, inject_layer, kernels_3x3)

    return model.to(DEVICE)


def inject_kernels(model, layer_name, kernels):
    """
    Inject kernels into the first conv of the specified ResNet-18 encoder layer.
    layer_name: 'layer2' or 'layer3'
    """
    # Access the target conv weight
    # ResNet-18 encoder layer structure: encoder.layer{N}[0].conv1.weight
    # Shape: (out_channels, in_channels, 3, 3)
    target = getattr(model.encoder, layer_name)
    conv_weight = target[0].conv1.weight  # (C_out, C_in, 3, 3)

    n_kernels = len(kernels)
    n_out = conv_weight.shape[0]

    print(f"Injecting {n_kernels} kernels into encoder.{layer_name}[0].conv1 "
          f"(shape: {conv_weight.shape})")

    with torch.no_grad():
        for i, k in enumerate(kernels):
            if i >= n_out:
                print(f"  Warning: more kernels than output channels, stopping at {i}")
                break
            k_tensor = torch.tensor(k, dtype=torch.float32)
            # Inject into first in_channel slot (mammography is single channel input by this layer too)
            conv_weight[i, 0, :, :] = k_tensor
            print(f"  Injected kernel {i} → channel {i}")


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    return ((2 * intersection + eps) / (pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) + eps)).mean()

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3)) - intersection
    return ((intersection + eps) / (union + eps)).mean()


def train_one_run(group_name, seed, inject_layer, train_loader, val_loader):
    model = build_model(seed, inject_layer)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    logger = setup_logger(group_name, seed)
    logger.info(f"Starting | group={group_name} | seed={seed} | inject_layer={inject_layer} | device={DEVICE}")
    logger.info(f"Epochs={EPOCHS} | LR={LR} | BatchSize={BATCH_SIZE}")
    logger.info("-" * 70)

    run = wandb.init(
        entity="rajshreerai931-abo-akademi",
        project="mammography-kernel-init",
        name=f"{group_name}_seed{seed}",
        config={
            "group": group_name,
            "seed": seed,
            "inject_layer": inject_layer,
            "epochs": EPOCHS,
            "lr": LR,
            "batch_size": BATCH_SIZE,
        },
        reinit=True
    )

    best_dice = 0.0
    early_dice = {}

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            preds = model(images)
            loss = criterion(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss, val_dice, val_iou = 0.0, 0.0, 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)
                preds = model(images)
                val_loss += criterion(preds, masks).item()
                val_dice += dice_score(preds, masks).item()
                val_iou += iou_score(preds, masks).item()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        scheduler.step()

        # Log early convergence checkpoints
        if epoch in [5, 10, 20]:
            early_dice[f"dice_epoch{epoch}"] = val_dice
            logger.info(f">>> CHECKPOINT ep{epoch} | Dice={val_dice:.4f}")

        # Save best checkpoint
        improved = ""
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(),
                       CHECKPOINT_DIR / f"{group_name}_seed{seed}_best.pt")
            improved = "  ★ best"

        # Live log line per epoch
        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_dice={val_dice:.4f} | "
            f"val_iou={val_iou:.4f}"
            f"{improved}"
        )

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "val_iou": val_iou,
        })

    logger.info("-" * 70)
    logger.info(f"Done | best_val_dice={best_dice:.4f}")
    logger.info(f"Early dice: {early_dice}")

    wandb.log(early_dice)
    wandb.log({"best_val_dice": best_dice})
    wandb.finish()

    return {"group": group_name, "seed": seed, "best_dice": best_dice, **early_dice}

def compute_kernel_drift(group_name, seed, inject_layer, kernels_3x3):
    """Compare injected kernel weights vs final trained weights."""
    model = build_model(seed=seed, inject_layer=None)  # fresh model, no injection
    model.load_state_dict(
        torch.load(CHECKPOINT_DIR / f"{group_name}_seed{seed}_best.pt",
                   map_location=DEVICE)
    )

    target = getattr(model.encoder, inject_layer)
    conv_weight = target[0].conv1.weight.detach().cpu().numpy()

    drifts = []
    for i, k_init in enumerate(kernels_3x3):
        if i >= conv_weight.shape[0]:
            break
        w_final = conv_weight[i, 0]
        drift = np.linalg.norm(w_final - k_init) / (np.linalg.norm(k_init) + 1e-8)
        drifts.append({"kernel_id": i, "drift": drift})
        print(f"  Kernel {i} drift: {drift:.4f}")

    return drifts

def main():

    # --- Phase 1 Conditions ---
    conditions = [
        ("A_baseline",      None),
        ("B_layer2_init",   "layer2"),
        ("C_layer3_init",   "layer3"),
    ]

    all_results = []

    for group_name, inject_layer in conditions:
        print(f"\n{'='*50}")
        print(f"Running group: {group_name}")
        for seed in SEEDS:
            print(f"  Seed: {seed}")
            train_loader, val_loader = make_loaders(seed)
            result = train_one_run(group_name, seed, inject_layer,
                                   train_loader, val_loader)
            all_results.append(result)

    # --- Summary ---
    import pandas as pd
    df = pd.DataFrame(all_results)
    summary = df.groupby("group").agg(["mean", "std"]).round(4)
    print("\n=== Phase 1 Summary ===")
    print(summary)
    summary.to_csv("phase1_results.csv")

    # --- Kernel drift for best group ---
    kernels_3x3 = get_3x3_kernels()
    print("\n=== Kernel Drift (Group B - layer2) ===")
    for seed in SEEDS:
        compute_kernel_drift("B_layer2_init", seed, "layer2", kernels_3x3)


if __name__ == "__main__":
    main()